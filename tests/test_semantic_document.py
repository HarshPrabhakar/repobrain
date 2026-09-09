from __future__ import annotations

from repobrain.models.retrieval import (
    ChunkType,
    CodeChunk,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    RelationshipType,
    ResolutionType,
    SymbolType,
)

from repobrain.retrieval.semantic_document import (
    SemanticCodeDocumentBuilder,
)


def make_hash_symbol() -> CodeSymbol:

    return CodeSymbol(
        symbol_id="sym_hash",

        repository_id="repo_test",

        file_id="file_scanner",

        name="_calculate_sha256",

        qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),

        fully_qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),

        symbol_type=SymbolType.METHOD,

        module=(
            "repobrain.ingestion.scanner"
        ),

        start_line=384,
        end_line=403,

        parent_symbol_id=(
            "sym_scanner"
        ),

        signature=(
            "_calculate_sha256("
            "path: Path, "
            "chunk_size: int = 1048576"
            ") -> str"
        ),

        docstring=(
            "Calculate SHA-256 without loading "
            "the entire file into memory."
        ),

        is_async=False,

        decorators=[],
    )


def make_chunk() -> CodeChunk:

    return CodeChunk(
        chunk_id="chunk_hash",

        repository_id="repo_test",

        file_id="file_scanner",

        relative_path=(
            "repobrain/ingestion/scanner.py"
        ),

        symbol_id="sym_hash",

        qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),

        chunk_type=ChunkType.METHOD,

        language="Python",

        start_line=384,
        end_line=403,

        text=(
            "def _calculate_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    digest.update(b'data')\n"
            "    return digest.hexdigest()\n"
        ),

        content_hash="hash_test",
    )


def make_call(
    relationship_id: str,
    target: str,
) -> CodeRelationship:

    return CodeRelationship(
        relationship_id=relationship_id,

        repository_id="repo_test",

        file_id="file_scanner",

        relationship_type=(
            RelationshipType.CALLS
        ),

        source_symbol_id="sym_hash",

        target_symbol_id=None,

        source_qualified_name=(
            "repobrain.ingestion.scanner."
            "RepositoryScanner."
            "_calculate_sha256"
        ),

        target_qualified_name=target,

        line_number=390,

        resolution=(
            ResolutionType.UNRESOLVED
        ),

        metadata={},
    )


def test_semantic_document_contains_symbol_identity() -> None:

    symbol = make_hash_symbol()

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[symbol],
            relationships=[],
        )
    )

    document = builder.build(
        make_chunk()
    )

    assert (
        "Symbol: _calculate_sha256"
        in document
    )

    assert (
        "Identifier terms: calculate sha256"
        in document
    )


def test_semantic_document_contains_docstring() -> None:

    symbol = make_hash_symbol()

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[symbol],
            relationships=[],
        )
    )

    document = builder.build(
        make_chunk()
    )

    assert (
        "Calculate SHA-256 without loading"
        in document
    )


def test_semantic_document_contains_signature() -> None:

    symbol = make_hash_symbol()

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[symbol],
            relationships=[],
        )
    )

    document = builder.build(
        make_chunk()
    )

    assert "Signature:" in document

    assert (
        "_calculate_sha256"
        in document
    )


def test_semantic_document_contains_calls() -> None:

    symbol = make_hash_symbol()

    relationships = [
        make_call(
            "rel_1",
            "hashlib.sha256",
        ),
        make_call(
            "rel_2",
            "digest.update",
        ),
        make_call(
            "rel_3",
            "digest.hexdigest",
        ),
    ]

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[symbol],
            relationships=relationships,
        )
    )

    document = builder.build(
        make_chunk()
    )

    assert "Calls:" in document
    assert "hashlib.sha256" in document
    assert "digest.update" in document
    assert "digest.hexdigest" in document


def test_semantic_document_contains_source() -> None:

    symbol = make_hash_symbol()

    chunk = make_chunk()

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[symbol],
            relationships=[],
        )
    )

    document = builder.build(
        chunk
    )

    assert "Source:" in document

    assert chunk.text in document


def test_chunk_without_symbol_uses_raw_text() -> None:

    chunk = CodeChunk(
        chunk_id="chunk_readme",

        repository_id="repo_test",

        file_id="file_readme",

        relative_path="README.md",

        symbol_id=None,

        qualified_name=None,

        chunk_type=ChunkType.FILE,

        language="Markdown",

        start_line=1,
        end_line=3,

        text=(
            "# RepoBrain\n"
            "Repository intelligence system."
        ),

        content_hash="hash_readme",
    )

    builder = (
        SemanticCodeDocumentBuilder(
            symbols=[],
            relationships=[],
        )
    )

    assert (
        builder.build(chunk)
        == chunk.text
    )