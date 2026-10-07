import pytest

from examples.multi_agent_rag import cosine_similarity, retrieve


def test_cosine_similarity_for_identical_vectors():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_retrieve_sorts_by_similarity():
    documents = [{"id": "match"}, {"id": "other"}]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    result = retrieve(documents, vectors, [0.9, 0.1], top_k=1)

    assert result[0]["id"] == "match"
    assert result[0]["score"] > 0.9


def test_retrieve_rejects_mismatched_document_embeddings():
    with pytest.raises(ValueError, match="exactly one embedding"):
        retrieve([{"id": "only"}], [], [1.0])
