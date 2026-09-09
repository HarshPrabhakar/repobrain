from __future__ import annotations

import hashlib

from pathlib import Path

from repobrain.models.repository import (
    FileMetadata,
    RepositoryScanResult,
)

from repobrain.models.retrieval import (
    ChunkType,
    CodeChunk,
)

from repobrain.models.symbols import (
    CodeSymbol,
    PythonRepositoryAnalysis,
    SymbolType,
)


class RepositoryChunkBuilder:
    """
    Build searchable repository chunks.

    Python:
        symbol-aware chunks using AST line boundaries.

    Non-Python:
        one whole-file chunk.

    The chunker performs no embeddings and no LLM processing.
    """

    def build(
        self,
        scan_result: RepositoryScanResult,
        analysis: PythonRepositoryAnalysis,
    ) -> list[CodeChunk]:

        symbols_by_file: dict[
            str,
            list[CodeSymbol],
        ] = {}

        for symbol in analysis.symbols:

            symbols_by_file.setdefault(
                symbol.file_id,
                [],
            ).append(symbol)

        chunks: list[CodeChunk] = []

        for file_metadata in scan_result.files:

            if file_metadata.language == "Python":

                chunks.extend(
                    self._build_python_chunks(
                        file_metadata=file_metadata,
                        symbols=symbols_by_file.get(
                            file_metadata.file_id,
                            [],
                        ),
                    )
                )

            else:

                chunk = self._build_file_chunk(
                    file_metadata
                )

                if chunk is not None:
                    chunks.append(chunk)

        return chunks

    # =========================================================
    # Python chunks
    # =========================================================

    def _build_python_chunks(
        self,
        file_metadata: FileMetadata,
        symbols: list[CodeSymbol],
    ) -> list[CodeChunk]:

        path = Path(
            file_metadata.absolute_path
        )

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lines = source.splitlines()

        chunks: list[CodeChunk] = []

        for symbol in symbols:

            # Module chunks currently contain the whole Python file.
            if symbol.symbol_type == SymbolType.MODULE:

                text = source

                if not text.strip():
                    continue

                chunks.append(
                    self._make_chunk(
                        file_metadata=file_metadata,
                        symbol=symbol,
                        chunk_type=ChunkType.MODULE,
                        start_line=1,
                        end_line=max(
                            len(lines),
                            1,
                        ),
                        text=text,
                    )
                )

                continue

            start_line = max(
                symbol.start_line,
                1,
            )

            end_line = min(
                symbol.end_line,
                len(lines),
            )

            if end_line < start_line:
                continue

            text = "\n".join(
                lines[
                    start_line - 1:
                    end_line
                ]
            )

            if not text.strip():
                continue

            chunks.append(
                self._make_chunk(
                    file_metadata=file_metadata,
                    symbol=symbol,
                    chunk_type=self._symbol_chunk_type(
                        symbol
                    ),
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                )
            )

        return chunks

    # =========================================================
    # Non-Python files
    # =========================================================

    def _build_file_chunk(
        self,
        file_metadata: FileMetadata,
    ) -> CodeChunk | None:

        path = Path(
            file_metadata.absolute_path
        )

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError:
            return None

        if not text.strip():
            return None

        line_count = max(
            len(text.splitlines()),
            1,
        )

        return self._make_chunk(
            file_metadata=file_metadata,
            symbol=None,
            chunk_type=ChunkType.FILE,
            start_line=1,
            end_line=line_count,
            text=text,
        )

    # =========================================================
    # Chunk creation
    # =========================================================

    def _make_chunk(
        self,
        file_metadata: FileMetadata,
        symbol: CodeSymbol | None,
        chunk_type: ChunkType,
        start_line: int,
        end_line: int,
        text: str,
    ) -> CodeChunk:

        qualified_name = (
            symbol.qualified_name
            if symbol is not None
            else None
        )

        symbol_id = (
            symbol.symbol_id
            if symbol is not None
            else None
        )

        chunk_id = self._chunk_id(
            repository_id=(
                file_metadata.repository_id
            ),
            file_id=file_metadata.file_id,
            symbol_id=symbol_id,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
        )

        content_hash = hashlib.sha256(
            text.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

        return CodeChunk(
            chunk_id=chunk_id,

            repository_id=(
                file_metadata.repository_id
            ),

            file_id=file_metadata.file_id,

            relative_path=(
                file_metadata.relative_path
            ),

            symbol_id=symbol_id,

            qualified_name=qualified_name,

            chunk_type=chunk_type,

            language=file_metadata.language,

            start_line=start_line,
            end_line=end_line,

            text=text,

            content_hash=content_hash,
        )

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _symbol_chunk_type(
        symbol: CodeSymbol,
    ) -> ChunkType:

        mapping = {
            SymbolType.CLASS:
                ChunkType.CLASS,

            SymbolType.FUNCTION:
                ChunkType.FUNCTION,

            SymbolType.METHOD:
                ChunkType.METHOD,

            SymbolType.MODULE:
                ChunkType.MODULE,
        }

        return mapping[
            symbol.symbol_type
        ]

    @staticmethod
    def _chunk_id(
        repository_id: str,
        file_id: str,
        symbol_id: str | None,
        chunk_type: ChunkType,
        start_line: int,
        end_line: int,
    ) -> str:

        value = (
            f"{repository_id}:"
            f"{file_id}:"
            f"{symbol_id}:"
            f"{chunk_type.value}:"
            f"{start_line}:"
            f"{end_line}"
        )

        digest = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

        return (
            f"chunk_{digest[:16]}"
        )