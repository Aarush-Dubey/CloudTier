"""Lease-based leader election backed by a single MongoDB document.

The optimizer must be able to run with several replicas for availability, but only
*one* replica may scan datasets and emit migration jobs at a time — otherwise the
replicas would create duplicate jobs for the same datasets. This module provides
mutual exclusion plus a monotonic **fencing token** using infrastructure the project
already depends on (Mongo), so it adds no new moving parts.

Design (see README "Leader election" section for the rationale vs. Kafka partition
ownership and full Raft/etcd):

* A ``leader_lock`` document holds ``holder_id``, ``lease_expires_at`` (a wall-clock
  deadline) and a ``fencing_token``.
* The token is bumped **only on acquisition** (a leadership change), never on renewal,
  so it is a monotonically increasing generation number for "who is in charge".
* Acquire and renew are each a single atomic ``find_one_and_update``. Renewal only
  matches the current holder; acquisition only matches an unheld or expired lease.
  Mongo executes each update atomically on the one document, so two replicas can never
  both win.

The class takes the locks collection and a ``clock`` callable, so it is fully unit
testable against the in-memory Mongo fake with no real Mongo or wall-clock sleeps.
"""

import os
import uuid
import time as _time

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from shared.config import Settings, settings


def default_instance_id() -> str:
    """Stable-ish identifier for this process/replica.

    Prefers an explicit ``INSTANCE_ID``/``HOSTNAME`` (Docker sets ``HOSTNAME`` to the
    container id), falling back to a random suffix so local runs still get a unique id.
    """
    return (
        os.getenv("INSTANCE_ID")
        or os.getenv("HOSTNAME")
        or f"optimizer-{uuid.uuid4().hex[:8]}"
    )


class LeaderElector:
    def __init__(
        self,
        locks_collection,
        lock_name: str = "optimizer",
        instance_id: str | None = None,
        lease_ttl_sec: float | None = None,
        config: Settings = settings,
        clock=_time.time,
    ):
        self.locks = locks_collection
        self.lock_name = lock_name
        self.instance_id = instance_id or config.instance_id or default_instance_id()
        self.lease_ttl_sec = (
            lease_ttl_sec if lease_ttl_sec is not None else config.leader_lease_ttl_sec
        )
        self._clock = clock
        self._is_leader = False
        self.fencing_token: int | None = None

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def ensure_lock(self) -> None:
        """Create the unique index and seed the lock document once (idempotent).

        Seeding the row up front keeps ``acquire`` out of the upsert path, so the hot
        loop is a plain conditional update on a single, guaranteed-present document.
        """
        self.locks.create_index([("lock_name", ASCENDING)], unique=True)
        try:
            self.locks.update_one(
                {"lock_name": self.lock_name},
                {
                    "$setOnInsert": {
                        "lock_name": self.lock_name,
                        "holder_id": None,
                        "lease_expires_at": 0.0,
                        "fencing_token": 0,
                    }
                },
                upsert=True,
            )
        except DuplicateKeyError:
            # Another replica seeded it first; that is exactly what we wanted.
            pass

    def _renew(self, now: float):
        # Only the current holder can renew; keeps the same fencing token.
        return self.locks.find_one_and_update(
            {"lock_name": self.lock_name, "holder_id": self.instance_id},
            {"$set": {"lease_expires_at": now + self.lease_ttl_sec, "renewed_at": now}},
            return_document=ReturnDocument.AFTER,
        )

    def _acquire(self, now: float):
        # Take over an unheld or expired lease; bump the fencing token (new generation).
        return self.locks.find_one_and_update(
            {
                "lock_name": self.lock_name,
                "$or": [
                    {"holder_id": None},
                    {"holder_id": {"$exists": False}},
                    {"lease_expires_at": {"$lte": now}},
                ],
            },
            {
                "$set": {
                    "holder_id": self.instance_id,
                    "lease_expires_at": now + self.lease_ttl_sec,
                    "acquired_at": now,
                    "renewed_at": now,
                },
                "$inc": {"fencing_token": 1},
            },
            return_document=ReturnDocument.AFTER,
        )

    def try_acquire(self, now: float | None = None) -> bool:
        """Renew if we already hold the lease, otherwise try to acquire it.

        Returns True iff this replica is the leader after the attempt. Safe to call on
        every heartbeat; call cadence must be comfortably shorter than the lease TTL.
        """
        now = self._clock() if now is None else now
        doc = self._renew(now)
        if doc is None:
            doc = self._acquire(now)
        leading = bool(doc and doc.get("holder_id") == self.instance_id)
        self._is_leader = leading
        self.fencing_token = doc.get("fencing_token") if doc else None
        return leading

    def release(self, now: float | None = None) -> None:
        """Voluntarily step down so another replica can take over immediately.

        Used on graceful shutdown and in tests. A crashed leader does *not* call this;
        in that case failover waits for the lease to expire, which is the behaviour the
        failover-time benchmark measures.
        """
        now = self._clock() if now is None else now
        self.locks.update_one(
            {"lock_name": self.lock_name, "holder_id": self.instance_id},
            {"$set": {"holder_id": None, "lease_expires_at": now - 1}},
        )
        self._is_leader = False
