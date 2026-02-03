from shared.leader import LeaderElector
from tests.fakes import FakeCollection


def make_elector(locks, instance_id, ttl=10.0):
    return LeaderElector(locks, instance_id=instance_id, lease_ttl_sec=ttl)


def test_acquire_then_renew_keeps_same_token():
    locks = FakeCollection()
    a = make_elector(locks, "a")
    a.ensure_lock()

    assert a.try_acquire(now=100) is True
    assert a.is_leader is True
    assert a.fencing_token == 1

    # Renewing before expiry keeps leadership and the same fencing token.
    assert a.try_acquire(now=103) is True
    assert a.fencing_token == 1


def test_only_one_replica_is_leader():
    locks = FakeCollection()
    a = make_elector(locks, "a")
    b = make_elector(locks, "b")
    a.ensure_lock()

    assert a.try_acquire(now=100) is True
    # B cannot acquire while A's lease is valid.
    assert b.try_acquire(now=101) is False
    assert b.is_leader is False
    # A still holds it.
    assert a.try_acquire(now=102) is True


def test_failover_after_lease_expiry_bumps_token():
    locks = FakeCollection()
    a = make_elector(locks, "a", ttl=10.0)
    b = make_elector(locks, "b", ttl=10.0)
    a.ensure_lock()

    assert a.try_acquire(now=100) is True
    assert a.fencing_token == 1

    # A's lease (expires at 110) is now expired; B takes over with a higher token.
    assert b.try_acquire(now=111) is True
    assert b.fencing_token == 2

    # A wakes up and discovers it is no longer the leader.
    assert a.try_acquire(now=112) is False
    assert a.is_leader is False


def test_fencing_token_is_monotonic_across_takeovers():
    locks = FakeCollection()
    a = make_elector(locks, "a", ttl=5.0)
    b = make_elector(locks, "b", ttl=5.0)
    a.ensure_lock()

    a.try_acquire(now=0)
    assert a.fencing_token == 1
    b.try_acquire(now=10)  # a expired at 5
    assert b.fencing_token == 2
    a.try_acquire(now=20)  # b expired at 15
    assert a.fencing_token == 3


def test_release_allows_immediate_takeover():
    locks = FakeCollection()
    a = make_elector(locks, "a", ttl=100.0)
    b = make_elector(locks, "b", ttl=100.0)
    a.ensure_lock()

    assert a.try_acquire(now=100) is True
    a.release(now=101)
    assert a.is_leader is False

    # Even though A's TTL was long, releasing lets B take over right away.
    assert b.try_acquire(now=102) is True
    assert b.fencing_token == 2


def test_ensure_lock_is_idempotent():
    locks = FakeCollection()
    a = make_elector(locks, "a")
    a.ensure_lock()
    a.ensure_lock()
    assert locks.count_documents({"lock_name": "optimizer"}) == 1
