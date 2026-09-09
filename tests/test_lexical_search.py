from __future__ import annotations

from repobrain.models.retrieval import (
    ChunkType,
    CodeChunk,
)

from repobrain.retrieval.lexical import (
    BM25Index,
    BM25Tokenizer,
)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    path: str = "app/main.py",
    qualified_name: str | None = None,
    language: str = "Python",
    chunk_type: ChunkType = ChunkType.FUNCTION,
) -> CodeChunk:

    return CodeChunk(
        chunk_id=chunk_id,

        repository_id="repo_test",

        file_id=f"file_{chunk_id}",

        relative_path=path,

        symbol_id=f"sym_{chunk_id}",

        qualified_name=qualified_name,

        chunk_type=chunk_type,

        language=language,

        start_line=1,
        end_line=10,

        text=text,

        content_hash="hash",
    )


def build_index() -> BM25Index:

    chunks = [
        make_chunk(
            "scanner",

            """
def scan(repository_path):
    root = validate_repository_root(repository_path)
    return root
""",

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner.scan"
            ),
        ),

        make_chunk(
            "sha",

            """
def calculate_sha256(path):
    digest = hashlib.sha256()
    return digest.hexdigest()
""",

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner._calculate_sha256"
            ),
        ),

        make_chunk(
            "ignore",

            """
def should_ignore_directory(path):
    if path.name in ignored_directories:
        return True
""",

            qualified_name=(
                "repobrain.ingestion.filters."
                "should_ignore_directory"
            ),
        ),

        make_chunk(
            "markdown",

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

    return BM25Index(
        chunks
    )


def test_tokenizer_keeps_identifier() -> None:

    tokenizer = BM25Tokenizer()

    tokens = tokenizer.tokenize(
        "calculate_sha256"
    )

    assert (
        "calculate_sha256"
        in tokens
    )


def test_tokenizer_splits_snake_case() -> None:

    tokenizer = BM25Tokenizer()

    tokens = tokenizer.tokenize(
        "calculate_sha256"
    )

    assert "calculate" in tokens

    assert "sha256" in tokens


def test_tokenizer_splits_camel_case() -> None:

    tokenizer = BM25Tokenizer()

    tokens = tokenizer.tokenize(
        "RepositoryScanner"
    )

    assert "repository" in tokens

    assert "scanner" in tokens


def test_exact_code_term_search() -> None:

    index = build_index()

    results = index.search(
        "sha256"
    )

    assert results

    assert (
        results[0].chunk_id
        == "sha"
    )


def test_multi_term_search() -> None:

    index = build_index()

    results = index.search(
        "calculate sha256"
    )

    assert results

    assert (
        results[0].chunk_id
        == "sha"
    )


def test_searches_qualified_name() -> None:

    index = build_index()

    results = index.search(
        "RepositoryScanner scan"
    )

    assert results

    assert (
        results[0].chunk_id
        == "scanner"
    )


def test_searches_source_text() -> None:

    index = build_index()

    results = index.search(
        "validate repository root"
    )

    assert results

    assert (
        results[0].chunk_id
        == "scanner"
    )


def test_searches_non_python_content() -> None:

    index = build_index()

    results = index.search(
        "agentic repository intelligence"
    )

    assert results

    assert (
        results[0].chunk_id
        == "markdown"
    )


def test_language_filter() -> None:

    index = build_index()

    results = index.search(
        "repository",
        language="Markdown",
    )

    assert results

    assert all(
        result.language
        == "Markdown"
        for result in results
    )


def test_unknown_query_returns_empty() -> None:

    index = build_index()

    results = index.search(
        "zzzznonexistentterm"
    )

    assert results == []


def test_empty_query_returns_empty() -> None:

    index = build_index()

    assert (
        index.search("   ")
        == []
    )


def test_top_k() -> None:

    index = build_index()

    results = index.search(
        "repository",
        top_k=1,
    )

    assert len(results) == 1


def test_zero_top_k_returns_empty() -> None:

    index = build_index()

    assert (
        index.search(
            "repository",
            top_k=0,
        )
        == []
    )


def test_matched_terms_are_returned() -> None:

    index = build_index()

    result = index.search(
        "calculate sha256"
    )[0]

    assert "calculate" in (
        result.matched_terms
    )

    assert "sha256" in (
        result.matched_terms
    )


def test_excerpt_is_returned() -> None:

    index = build_index()

    result = index.search(
        "ignored directories"
    )[0]

    assert result.excerpt