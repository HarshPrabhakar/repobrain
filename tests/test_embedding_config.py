from repobrain.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    QUERY_PREFIX,
)


def test_default_embedding_model_is_code_specialized() -> None:

    assert (
        DEFAULT_EMBEDDING_MODEL
        == "nomic-ai/CodeRankEmbed"
    )


def test_fallback_embedding_model_is_available() -> None:

    assert (
        FALLBACK_EMBEDDING_MODEL
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_code_search_query_prefix() -> None:

    assert (
        QUERY_PREFIX
        == (
            "Represent this query "
            "for searching relevant code: "
        )
    )