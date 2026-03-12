"""
Test Embeddings — Test BaseEmbedding interface bằng mock subclass.

Không load model thật (Voyage4Nano nặng), dùng fake embedding để test logic.
"""

from data.embeddings.base_embedding import BaseEmbedding


# ------------------------------------------------------------------
# Fake Embedding — mock subclass cho testing
# ------------------------------------------------------------------

class FakeEmbedding(BaseEmbedding):
    """
    Mock embedding: trả vector cố định [0.1, 0.2, ..., 0.1] có đúng dimensions chiều.
    Dùng cho testing mà không cần load model thật.
    """

    def __init__(self, dimensions: int = 4) -> None:
        super().__init__(model_name="fake-model", dimensions=dimensions)

    def _embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            return [0.1 * (i + 1) for i in range(self._dimensions)]
        return [
            [0.1 * (i + 1) for i in range(self._dimensions)]
            for _ in texts
        ]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestBaseEmbedding:
    """Test BaseEmbedding interface qua FakeEmbedding."""

    def test_model_name(self):
        emb = FakeEmbedding()
        assert emb.model_name == "fake-model"

    def test_dimensions(self):
        emb = FakeEmbedding(dimensions=8)
        assert emb.dimensions == 8

    def test_embed_documents_returns_list_of_vectors(self):
        emb = FakeEmbedding(dimensions=4)
        result = emb.embed_documents(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 4

    def test_embed_documents_empty_input(self):
        emb = FakeEmbedding()
        result = emb.embed_documents([])
        assert result == []

    def test_embed_query_returns_single_vector(self):
        emb = FakeEmbedding(dimensions=4)
        result = emb.embed_query("search query")

        assert isinstance(result, list)
        assert len(result) == 4

    def test_embed_documents_consistent_dimensions(self):
        """Mọi vector phải có cùng số chiều."""
        dims = 16
        emb = FakeEmbedding(dimensions=dims)
        result = emb.embed_documents(["a", "b", "c"])

        for vec in result:
            assert len(vec) == dims

    def test_repr(self):
        emb = FakeEmbedding(dimensions=4)
        r = repr(emb)
        assert "FakeEmbedding" in r
        assert "fake-model" in r
        assert "4" in r
