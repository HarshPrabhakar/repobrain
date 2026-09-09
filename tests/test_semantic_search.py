from __future__ import annotations

import numpy as np

from repobrain.embeddings.base import (
    EmbeddingProvider,
)

from repobrain.models.retrieval import (
    ChunkType,
    CodeChunk,
)

from repobrain.retrieval.semantic import (
    SemanticSearchEngine,
)


class FakeEmbeddingProvider(
    EmbeddingProvider
):
    """
    Deterministic fake provider for semantic-search tests.

    Dimensions:

        0 -> repository/path security
        1 -> hashing/checksums
        2 -> documentation/project description
        3 -> directory filtering
    """

    @property
    def dimension(
        self,
    ) -> int:

        return 4

    @property
    def model_name(
        self,
    ) -> str:

        return "fake-test-embedding"

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        vectors = [
            self._vector(text)
            for text in texts
        ]

        if not vectors:

            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:

        return np.asarray(
            self._vector(text),
            dtype=np.float32,
        )

    def _vector(
        self,
        text: str,
    ) -> list[float]:

        value = text.casefold()

        vector = np.zeros(
            self.dimension,
            dtype=np.float32,
        )

        security_terms = (
            "unsafe",
            "escape",
            "boundary",
            "outside",
            "path security",
            "leaving the repository",
        )

        hash_terms = (
            "sha256",
            "checksum",
            "hash",
            "digest",
            "fingerprint",
        )

        documentation_terms = (
            "agentic",
            "intelligence system",
            "documentation",
            "project description",
            "readme",
        )

        filtering_terms = (
            "ignored directory",
            "ignore directory",
            "directory filtering",
            "skip folders",
            "excluded directories",
        )

        if any(
            term in value
            for term in security_terms
        ):
            vector[0] += 1.0

        if any(
            term in value
            for term in hash_terms
        ):
            vector[1] += 1.0

        if any(
            term in value
            for term in documentation_terms
        ):
            vector[2] += 1.0

        if any(
            term in value
            for term in filtering_terms
        ):
            vector[3] += 1.0

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm == 0.0:

            # Deterministic fallback direction.
            vector[:] = 0.5

            norm = float(
                np.linalg.norm(
                    vector
                )
            )

        vector /= norm

        return vector.tolist()


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    path: str,
    qualified_name: str | None,
    language: str = "Python",
    chunk_type: ChunkType = ChunkType.METHOD,
) -> CodeChunk:

    return CodeChunk(
        chunk_id=chunk_id,

        repository_id="repo_test",

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

        language=language,

        start_line=1,
        end_line=10,

        text=text,

        content_hash=(
            f"hash_{chunk_id}"
        ),
    )


def build_engine() -> SemanticSearchEngine:

    chunks = [
        make_chunk(
            "security",

            """
Prevent a resolved path from escaping
the repository boundary.
Raise PermissionError when the path
is outside the repository root.
""",

            path=(
                "repobrain/"
                "ingestion/scanner.py"
            ),

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner."
                "_ensure_inside_repository"
            ),
        ),

        make_chunk(
            "hash",

            """
Calculate a SHA256 digest for a file
without loading the entire file into memory.
""",

            path=(
                "repobrain/"
                "ingestion/scanner.py"
            ),

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner."
                "_calculate_sha256"
            ),
        ),

        make_chunk(
            "filter",

            """
Ignore excluded directories and prevent
walking into ignored directory trees.
""",

            path=(
                "repobrain/"
                "ingestion/filters.py"
            ),

            qualified_name=(
                "repobrain.ingestion.filters."
                "should_ignore_directory"
            ),
        ),

        make_chunk(
            "readme",

            """
RepoBrain is an agentic repository
intelligence system.
""",

            path="README.md",

            qualified_name=None,

            language="Markdown",

            chunk_type=ChunkType.FILE,
        ),
    ]

    return SemanticSearchEngine(
        chunks,
        FakeEmbeddingProvider(),
    )


def test_index_contains_chunks() -> None:

    engine = build_engine()

    assert len(engine) == 4


def test_embedding_dimension() -> None:

    engine = build_engine()

    assert engine.dimension == 4


def test_model_name() -> None:

    engine = build_engine()

    assert (
        engine.model_name
        == "fake-test-embedding"
    )


def test_semantic_security_query() -> None:

    engine = build_engine()

    results = engine.search(
        (
            "where do we stop unsafe paths "
            "from leaving the repository?"
        )
    )

    assert results

    assert (
        results[0].chunk_id
        == "security"
    )


def test_semantic_hash_query() -> None:

    engine = build_engine()

    results = engine.search(
        "where is the file checksum created?"
    )

    assert results

    assert (
        results[0].chunk_id
        == "hash"
    )


def test_semantic_directory_filter_query() -> None:

    engine = build_engine()

    results = engine.search(
        "how do we skip folders?"
    )

    assert results

    assert (
        results[0].chunk_id
        == "filter"
    )


def test_semantic_documentation_query() -> None:

    engine = build_engine()

    results = engine.search(
        "where is the project description?"
    )

    assert results

    assert (
        results[0].chunk_id
        == "readme"
    )


def test_language_filter() -> None:

    engine = build_engine()

    results = engine.search(
        "project description",
        language="Markdown",
    )

    assert results

    assert all(
        result.language
        == "Markdown"
        for result in results
    )


def test_python_language_filter() -> None:

    engine = build_engine()

    results = engine.search(
        "unsafe path",
        language="Python",
    )

    assert results

    assert all(
        result.language
        == "Python"
        for result in results
    )


def test_top_k() -> None:

    engine = build_engine()

    results = engine.search(
        "repository",
        top_k=2,
    )

    assert len(results) == 2


def test_zero_top_k() -> None:

    engine = build_engine()

    assert (
        engine.search(
            "repository",
            top_k=0,
        )
        == []
    )


def test_empty_query() -> None:

    engine = build_engine()

    assert (
        engine.search("   ")
        == []
    )


def test_empty_index() -> None:

    engine = SemanticSearchEngine(
        [],
        FakeEmbeddingProvider(),
    )

    assert len(engine) == 0

    assert (
        engine.search(
            "anything"
        )
        == []
    )


def test_result_contains_source_location() -> None:

    engine = build_engine()

    result = engine.search(
        "file checksum"
    )[0]

    assert result.relative_path

    assert result.start_line == 1

    assert result.end_line == 10


def test_result_contains_excerpt() -> None:

    engine = build_engine()

    result = engine.search(
        "file checksum"
    )[0]

    assert result.excerpt


def test_min_score_filter() -> None:

    engine = build_engine()

    results = engine.search(
        "file checksum",
        min_score=0.99,
    )

    assert results

    assert all(
        result.score >= 0.99
        for result in results
    )