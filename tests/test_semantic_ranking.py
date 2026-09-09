from __future__ import annotations

from repobrain.models.retrieval import (
    ChunkType,
    SemanticSearchResult,
)

from repobrain.retrieval.semantic_ranking import (
    SemanticIntentClassifier,
    SemanticQueryIntent,
    SemanticRankingPolicy,
    SemanticSourceClassifier,
    SemanticSourceKind,
)


def make_result(
    *,
    chunk_id: str,
    path: str,
    chunk_type: ChunkType,
    score: float,
    qualified_name: str | None = None,
) -> SemanticSearchResult:

    return SemanticSearchResult(
        chunk_id=chunk_id,

        file_id=(
            f"file_{chunk_id}"
        ),

        relative_path=path,

        symbol_id=(
            f"sym_{chunk_id}"
            if qualified_name
            else None
        ),

        qualified_name=(
            qualified_name
        ),

        chunk_type=chunk_type,

        language="Python",

        start_line=1,
        end_line=10,

        score=score,

        excerpt="example",
    )


def test_detects_implementation_intent() -> None:

    assert (
        SemanticIntentClassifier.classify(
            (
                "which function computes "
                "a stable digest?"
            )
        )
        == SemanticQueryIntent.IMPLEMENTATION
    )


def test_detects_documentation_intent() -> None:

    assert (
        SemanticIntentClassifier.classify(
            (
                "what is this project "
                "trying to build?"
            )
        )
        == SemanticQueryIntent.DOCUMENTATION
    )


def test_detects_general_intent() -> None:

    assert (
        SemanticIntentClassifier.classify(
            "repository intelligence"
        )
        == SemanticQueryIntent.GENERAL
    )


def test_classifies_test_source() -> None:

    assert (
        SemanticSourceClassifier.classify(
            "tests/test_scanner.py"
        )
        == SemanticSourceKind.TEST
    )


def test_classifies_production_source() -> None:

    assert (
        SemanticSourceClassifier.classify(
            (
                "repobrain/"
                "ingestion/scanner.py"
            )
        )
        == SemanticSourceKind.PRODUCTION
    )


def test_classifies_documentation_source() -> None:

    assert (
        SemanticSourceClassifier.classify(
            "README.md"
        )
        == SemanticSourceKind.DOCUMENTATION
    )


def test_classifies_config_source() -> None:

    assert (
        SemanticSourceClassifier.classify(
            "pyproject.toml"
        )
        == SemanticSourceKind.CONFIG
    )


def test_implementation_promotes_production_method() -> None:

    test_result = make_result(
        chunk_id="test",
        path="tests/test_scanner.py",
        chunk_type=ChunkType.FUNCTION,
        score=0.27,
        qualified_name=(
            "tests.test_scanner."
            "test_content_hash"
        ),
    )

    implementation = make_result(
        chunk_id="sha",
        path=(
            "repobrain/"
            "ingestion/scanner.py"
        ),
        chunk_type=ChunkType.METHOD,
        score=0.20,
        qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),
    )

    policy = (
        SemanticRankingPolicy()
    )

    results = policy.rerank(
        query=(
            "which function computes "
            "a stable digest?"
        ),
        results=[
            test_result,
            implementation,
        ],
        top_k=2,
    )

    assert (
        results[0].chunk_id
        == "sha"
    )

    assert (
        results[0].ranking_score
        is not None
    )

    assert (
        results[0].score
        == 0.20
    )


def test_documentation_promotes_readme() -> None:

    readme = make_result(
        chunk_id="readme",
        path="README.md",
        chunk_type=ChunkType.FILE,
        score=0.30,
    )

    implementation = make_result(
        chunk_id="method",
        path=(
            "repobrain/"
            "ingestion/scanner.py"
        ),
        chunk_type=ChunkType.METHOD,
        score=0.33,
        qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner.scan"
        ),
    )

    results = (
        SemanticRankingPolicy()
        .rerank(
            query=(
                "what is this project "
                "trying to build?"
            ),
            results=[
                implementation,
                readme,
            ],
            top_k=2,
        )
    )

    assert (
        results[0].chunk_id
        == "readme"
    )


def test_raw_semantic_score_is_preserved() -> None:

    result = make_result(
        chunk_id="implementation",
        path=(
            "repobrain/"
            "ingestion/scanner.py"
        ),
        chunk_type=ChunkType.METHOD,
        score=0.1989,
        qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),
    )

    reranked = (
        SemanticRankingPolicy()
        .rerank(
            query=(
                "where is the checksum "
                "calculated?"
            ),
            results=[result],
            top_k=1,
        )
    )

    assert (
        reranked[0].score
        == 0.1989
    )

    assert (
        reranked[0].ranking_score
        != reranked[0].score
    )