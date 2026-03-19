"""
Test Documents — Test QdrantDocumentStore + BaseDocumentStore.

Dùng QdrantClient(":memory:") để test in-memory, không tạo file trên disk.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct

from src.data.documents.qdrant_store import QdrantDocumentStore


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

VECTOR_SIZE = 4  # Nhỏ cho nhanh


def _make_store(collection_name: str = "test_collection") -> QdrantDocumentStore:
    """Tạo store in-memory cho testing."""
    # Reset singleton để mỗi test có fresh instance
    QdrantDocumentStore._instance = None

    client = QdrantClient(":memory:")
    return QdrantDocumentStore(
        collection_name=collection_name,
        qdrant_client=client,
        vector_size=VECTOR_SIZE,
        distance=Distance.COSINE,
    )


def _make_point(id: int | str, text: str = "sample") -> PointStruct:
    """Tạo PointStruct giả với vector cố định."""
    return PointStruct(
        id=id,
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"text": text, "tenant_id": "tenant-1"},
    )


# ------------------------------------------------------------------
# Test Collection Management
# ------------------------------------------------------------------

class TestCollectionManagement:
    """Test create, delete, list collections."""

    def test_create_collection_on_init(self):
        """Collection mặc định phải được tạo khi init."""
        store = _make_store("auto_created")
        assert "auto_created" in store.list_collections()

    def test_create_collection_idempotent(self):
        """Gọi create 2 lần không bị lỗi."""
        store = _make_store("my_col")
        store.create_collection("my_col")  # Lần 2
        assert store.list_collections().count("my_col") == 1

    def test_create_another_collection(self):
        """Tạo thêm collection mới ngoài collection mặc định."""
        store = _make_store("default")
        store.create_collection("extra")
        collections = store.list_collections()
        assert "default" in collections
        assert "extra" in collections

    def test_delete_collection(self):
        store = _make_store("to_delete")
        assert "to_delete" in store.list_collections()

        store.delete_collection("to_delete")
        assert "to_delete" not in store.list_collections()

    def test_delete_nonexistent_collection(self):
        """Xóa collection không tồn tại không bị lỗi."""
        store = _make_store("default")
        store.delete_collection("nonexistent")  # Should not raise

    def test_list_collections(self):
        store = _make_store("col_a")
        store.create_collection("col_b")
        collections = store.list_collections()
        assert "col_a" in collections
        assert "col_b" in collections


# ------------------------------------------------------------------
# Test Document Operations
# ------------------------------------------------------------------

class TestDocumentOperations:
    """Test upsert + delete documents."""

    def test_upsert_single_point(self):
        store = _make_store()
        point = _make_point(1, "hello world")

        store.upsert_documents(points=[point])

        # Verify bằng search
        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )
        assert len(results) == 1
        assert results[0].payload["text"] == "hello world"

    def test_upsert_multiple_points(self):
        store = _make_store()
        points = [
            _make_point(1, "first"),
            _make_point(2, "second"),
            _make_point(3, "third"),
        ]

        store.upsert_documents(points=points)

        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=10,
        )
        assert len(results) == 3

    def test_upsert_overwrites_existing(self):
        """Upsert cùng ID phải update, không duplicate."""
        store = _make_store()

        store.upsert_documents(points=[_make_point(1, "old text")])
        store.upsert_documents(points=[_make_point(1, "new text")])

        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=10,
        )
        assert len(results) == 1
        assert results[0].payload["text"] == "new text"

    def test_upsert_to_different_collection(self):
        store = _make_store("default")
        store.create_collection("other")

        store.upsert_documents(
            points=[_make_point(1, "in other")],
            collection_name="other",
        )

        # Default collection should be empty
        results_default = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            collection_name="default",
            limit=10,
        )
        assert len(results_default) == 0

        # Other collection should have the point
        results_other = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            collection_name="other",
            limit=10,
        )
        assert len(results_other) == 1

    def test_delete_documents(self):
        store = _make_store()
        store.upsert_documents(points=[_make_point(1), _make_point(2)])

        store.delete_documents(ids=[1])

        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=10,
        )
        assert len(results) == 1


# ------------------------------------------------------------------
# Test Search
# ------------------------------------------------------------------

class TestSearch:
    """Test search cơ bản."""

    def test_search_returns_results(self):
        store = _make_store()
        store.upsert_documents(points=[
            _make_point(1, "hello"),
            _make_point(2, "world"),
        ])

        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )
        assert len(results) == 2

    def test_search_respects_limit(self):
        store = _make_store()
        points = [_make_point(i, f"text-{i}") for i in range(10)]
        store.upsert_documents(points=points)

        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=3,
        )
        assert len(results) == 3

    def test_search_empty_collection(self):
        store = _make_store()
        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )
        assert len(results) == 0

    def test_search_with_filter(self):
        """Test search có filter trên payload."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        store = _make_store()
        store.upsert_documents(points=[
            PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4],
                        payload={"text": "a", "tenant_id": "t1"}),
            PointStruct(id=2, vector=[0.1, 0.2, 0.3, 0.4],
                        payload={"text": "b", "tenant_id": "t2"}),
            PointStruct(id=3, vector=[0.1, 0.2, 0.3, 0.4],
                        payload={"text": "c", "tenant_id": "t1"}),
        ])

        f = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value="t1"))]
        )
        results = store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            query_filter=f,
            limit=10,
        )
        assert len(results) == 2
        for r in results:
            assert r.payload["tenant_id"] == "t1"


# ------------------------------------------------------------------
# Test Properties
# ------------------------------------------------------------------

class TestProperties:
    """Test properties và __repr__."""

    def test_collection_name_property(self):
        store = _make_store("my_collection")
        assert store.collection_name == "my_collection"

    def test_client_property(self):
        store = _make_store()
        assert isinstance(store.client, QdrantClient)

    def test_repr(self):
        store = _make_store("test_repr")
        r = repr(store)
        assert "QdrantDocumentStore" in r
        assert "test_repr" in r
