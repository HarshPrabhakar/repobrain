from __future__ import annotations

from collections.abc import Iterable

import faiss
import numpy as np

from repobrain.embeddings.base import (
    EmbeddingProvider,
)

from repobrain.models.retrieval import (
    CodeChunk,
    SemanticSearchResult,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
)

from repobrain.retrieval.semantic_document import (
    SemanticCodeDocumentBuilder,
)


class SemanticSearchEngine:
    """
    RepoBrain semantic repository search.

    Phase 3C:
        vector similarity using FAISS

    Phase 3C.2:
        deterministic AST-aware semantic document construction

    Architecture:

        CodeChunk
            +
        CodeSymbol
            +
        CodeRelationship
            ↓
        SemanticCodeDocumentBuilder
            ↓
        EmbeddingProvider
            ↓
        normalized vectors
            ↓
        FAISS IndexFlatIP
    """

    def __init__(
        self,
        chunks: list[CodeChunk],
        embedding_provider: EmbeddingProvider,
        *,
        symbols: Iterable[CodeSymbol] | None = None,
        relationships: Iterable[
            CodeRelationship
        ] | None = None,
    ) -> None:

        self.chunks = list(
            chunks
        )

        self.embedding_provider = (
            embedding_provider
        )

        self._dimension = (
            embedding_provider.dimension
        )

        self._semantic_document_builder: (
            SemanticCodeDocumentBuilder
            | None
        ) = None

        if symbols is not None:

            self._semantic_document_builder = (
                SemanticCodeDocumentBuilder(
                    symbols=symbols,
                    relationships=(
                        relationships
                        if relationships
                        is not None
                        else []
                    ),
                )
            )

        self._index = (
            faiss.IndexFlatIP(
                self._dimension
            )
        )

        self._build()

    # =========================================================
    # Index construction
    # =========================================================

    def _build(
        self,
    ) -> None:

        if not self.chunks:
            return

        texts = [
            self._embedding_text(
                chunk
            )
            for chunk
            in self.chunks
        ]

        embeddings = (
            self.embedding_provider
            .embed_documents(
                texts
            )
        )

        embeddings = (
            self._validate_embeddings(
                embeddings,
                expected_rows=len(
                    self.chunks
                ),
            )
        )

        self._index.add(
            embeddings
        )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        language: str | None = None,
        min_score: float | None = None,
    ) -> list[SemanticSearchResult]:

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            return []

        if top_k <= 0:
            return []

        if not self.chunks:
            return []

        query_embedding = (
            self.embedding_provider
            .embed_query(
                normalized_query
            )
        )

        query_embedding = (
            self._validate_query_embedding(
                query_embedding
            )
        )

        if language is None:

            search_k = min(
                top_k,
                len(self.chunks),
            )

        else:

            # Search every candidate before applying language
            # filtering so valid filtered results cannot be
            # accidentally excluded.
            search_k = len(
                self.chunks
            )

        distances, indices = (
            self._index.search(
                query_embedding.reshape(
                    1,
                    -1,
                ),
                search_k,
            )
        )

        results: list[
            SemanticSearchResult
        ] = []

        for score, index_value in zip(
            distances[0],
            indices[0],
            strict=True,
        ):

            document_index = int(
                index_value
            )

            if document_index < 0:
                continue

            if document_index >= len(
                self.chunks
            ):
                continue

            chunk = self.chunks[
                document_index
            ]

            if (
                language is not None
                and chunk.language
                != language
            ):
                continue

            numeric_score = float(
                score
            )

            # Protect against very small numerical overshoots.
            numeric_score = max(
                -1.0,
                min(
                    1.0,
                    numeric_score,
                ),
            )

            if (
                min_score is not None
                and numeric_score
                < min_score
            ):
                continue

            results.append(
                SemanticSearchResult(
                    chunk_id=(
                        chunk.chunk_id
                    ),

                    file_id=(
                        chunk.file_id
                    ),

                    relative_path=(
                        chunk.relative_path
                    ),

                    symbol_id=(
                        chunk.symbol_id
                    ),

                    qualified_name=(
                        chunk.qualified_name
                    ),

                    chunk_type=(
                        chunk.chunk_type
                    ),

                    language=(
                        chunk.language
                    ),

                    start_line=(
                        chunk.start_line
                    ),

                    end_line=(
                        chunk.end_line
                    ),

                    score=(
                        numeric_score
                    ),

                    excerpt=self._excerpt(
                        chunk.text
                    ),
                )
            )

            if (
                len(results)
                >= top_k
            ):
                break

        return results

    # =========================================================
    # Semantic representation
    # =========================================================

    def _embedding_text(
        self,
        chunk: CodeChunk,
    ) -> str:
        """
        Return text that will actually be embedded.

        When Phase 3C.2 repository intelligence is supplied,
        build an AST-aware deterministic semantic document.

        Otherwise preserve the Phase 3C raw-code behavior.
        """

        if (
            self._semantic_document_builder
            is not None
        ):

            return (
                self._semantic_document_builder
                .build(
                    chunk
                )
            )

        return self._searchable_text(
            chunk
        )

    @staticmethod
    def _searchable_text(
        chunk: CodeChunk,
    ) -> str:
        """
        Phase 3C fallback representation.

        Keep this method because existing tests and callers
        rely on raw-code semantic search when no repository
        symbol intelligence is supplied.
        """

        return chunk.text

    # =========================================================
    # Embedding validation
    # =========================================================

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        expected_rows: int,
    ) -> np.ndarray:

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim != 2:

            raise ValueError(
                "Document embeddings must be "
                "a 2-dimensional array."
            )

        if (
            array.shape[0]
            != expected_rows
        ):

            raise ValueError(
                "Embedding row count does not "
                "match chunk count."
            )

        if (
            array.shape[1]
            != self._dimension
        ):

            raise ValueError(
                "Embedding dimension does not "
                "match provider dimension."
            )

        return np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    def _validate_query_embedding(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:

        array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if array.ndim != 1:

            raise ValueError(
                "Query embedding must be "
                "one-dimensional."
            )

        if (
            array.shape[0]
            != self._dimension
        ):

            raise ValueError(
                "Query embedding dimension does "
                "not match index dimension."
            )

        return np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    # =========================================================
    # Excerpts
    # =========================================================

    @staticmethod
    def _excerpt(
        text: str,
        *,
        max_length: int = 280,
    ) -> str:

        collapsed = " ".join(
            text.split()
        )

        if (
            len(collapsed)
            <= max_length
        ):

            return collapsed

        return (
            collapsed[
                :max_length
            ]
            + "..."
        )

    # =========================================================
    # Metadata
    # =========================================================

    @property
    def dimension(
        self,
    ) -> int:

        return self._dimension

    @property
    def model_name(
        self,
    ) -> str:

        return (
            self.embedding_provider
            .model_name
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.chunks
        )