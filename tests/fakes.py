import itertools
import time


class UpdateResult:
    def __init__(self, upserted_id=None):
        self.upserted_id = upserted_id


class InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class QueryResult(list):
    def sort(self, key, direction=-1):
        return QueryResult(sorted(self, key=lambda doc: doc.get(key, 0), reverse=direction < 0))

    def limit(self, count):
        return QueryResult(self[:count])


class FakeCollection:
    _ids = itertools.count(1)

    def __init__(self, docs=None):
        self.docs = []
        for doc in docs or []:
            self.insert(doc.copy())

    def insert(self, doc):
        doc.setdefault("_id", next(self._ids))
        self.docs.append(doc)
        return doc["_id"]

    def insert_one(self, doc):
        return InsertResult(self.insert(doc))

    def create_index(self, *args, **kwargs):
        return None

    def _matches(self, doc, query):
        for key, value in query.items():
            if key == "$or":
                if not any(self._matches(doc, item) for item in value):
                    return False
                continue
            actual, exists = self._lookup(doc, key)
            if isinstance(value, dict):
                if "$exists" in value and exists != value["$exists"]:
                    return False
                if "$lte" in value and actual > value["$lte"]:
                    return False
                continue
            if actual != value:
                return False
        return True

    def _lookup(self, doc, key):
        current = doc
        for part in key.split("."):
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    return None, False
                current = current[index]
                continue
            if not isinstance(current, dict) or part not in current:
                return None, False
            current = current[part]
        return current, True

    def find_one_and_update(self, query, update, upsert=False, return_document=None, sort=None):
        matches = [doc for doc in self.docs if self._matches(doc, query)]
        if sort:
            for key, direction in reversed(sort):
                matches.sort(key=lambda doc: doc.get(key, 0), reverse=direction < 0)
        doc = matches[0] if matches else None
        if not doc and upsert:
            doc = {key: value for key, value in query.items() if not key.startswith("$")}
            self.insert(doc)
        if not doc:
            return None
        self._apply(doc, update, inserting=False)
        return doc.copy()

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if self._matches(doc, query):
                self._apply(doc, update, inserting=False)
                return UpdateResult()
        if upsert:
            doc = {key: value for key, value in query.items() if not key.startswith("$")}
            self._apply(doc, update, inserting=True)
            inserted_id = self.insert(doc)
            return UpdateResult(inserted_id)
        return UpdateResult()

    def _apply(self, doc, update, inserting):
        if "$set" in update:
            doc.update(update["$set"])
        if inserting and "$setOnInsert" in update:
            doc.update(update["$setOnInsert"])
        if "$inc" in update:
            for key, value in update["$inc"].items():
                doc[key] = doc.get(key, 0) + value
        if "$push" in update:
            for key, spec in update["$push"].items():
                values = spec.get("$each", [spec])
                doc.setdefault(key, []).extend(values)
                if "$slice" in spec:
                    doc[key] = doc[key][spec["$slice"] :]

    def count_documents(self, query):
        return len([doc for doc in self.docs if self._matches(doc, query)])

    def find(self, query=None):
        query = query or {}
        return QueryResult([doc.copy() for doc in self.docs if self._matches(doc, query)])

    def find_one(self, query=None, sort=None):
        docs = self.find(query)
        if sort:
            for key, direction in reversed(sort):
                docs = docs.sort(key, direction)
        return docs[0].copy() if docs else None

    def aggregate(self, pipeline):
        if pipeline == [{"$group": {"_id": "$current_backend", "count": {"$sum": 1}}}]:
            counts = {}
            for doc in self.docs:
                counts[doc.get("current_backend")] = counts.get(doc.get("current_backend"), 0) + 1
            return [{"_id": key, "count": value} for key, value in counts.items()]
        return []


def sample_event(dataset_id="ds_000001", reads=150, backend="public-cold"):
    return {
        "timestamp": int(time.time()),
        "dataset_id": dataset_id,
        "reads_1h": reads,
        "writes_1h": 2,
        "bytes_read_1h": 1024,
        "hour_of_day": 12,
        "day_of_week": 2,
        "current_backend": backend,
        "initial_backend": backend,
        "size_gb": 25.0,
    }
