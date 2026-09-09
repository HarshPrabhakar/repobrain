from __future__ import annotations

import numpy as np
import faiss

from repobrain.embeddings.base import (
    EmbeddingProvider,
)

from repobrain.models.retrieval import (
    CodeChunk,
    SemanticSearchResult,
)


class SemanticSearchEngine:
    """
    Phase 3C semantic repository search.

    Architecture:

        CodeChunk
            ↓
        EmbeddingProvider
            ↓
        normalized vectors
            ↓
        FAISS IndexFlatIP
            ↓
        semantic nearest-neighbor search

    The index is currently in-memory.

    Persistence will be added when RepoBrain's storage layer
    is introduced.
    """

    def __init__(
        self,
        chunks: list[CodeChunk],
        embedding_provider: EmbeddingProvider,
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

        self._index = faiss.IndexFlatIP(
            self._dimension
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
            self._searchable_text(chunk)
            for chunk in self.chunks
        ]

        embeddings = (
            self.embedding_provider
            .embed_documents(
                texts
            )
        )

        embeddings = self._validate_embeddings(
            embeddings,
            expected_rows=len(
                self.chunks
            ),
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
        """
        Perform semantic nearest-neighbor search.

        When a language filter is provided, all indexed
        candidates are retrieved before filtering. This keeps
        filtering correct for our current small repository scale.
        """

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

            # Search all candidates first so post-search
            # filtering cannot accidentally hide valid
            # language-specific results.
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
                and chunk.language != language
            ):
                continue

            numeric_score = float(
                score
            )

            # Floating-point operations can produce tiny
            # numerical overshoots such as 1.0000001.
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
                    chunk_id=chunk.chunk_id,

                    file_id=chunk.file_id,

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

                    score=numeric_score,

                    excerpt=self._excerpt(
                        chunk.text
                    ),
                )
            )

            if len(results) >= top_k:
                break

        return results

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
    # Search text
    # =========================================================

    @staticmethod
    def _searchable_text(
        chunk: CodeChunk,
    ) -> str:
        """
        Return the semantic document representation.

        Phase 3C.1 intentionally embeds the raw chunk text.

        Code-specialized retrieval models such as CodeRankEmbed
        are trained to compare an instructed natural-language
        query against source code directly.

        Structural metadata such as file path, language, chunk
        type, and qualified symbol name remains available on the
        CodeChunk and SemanticSearchResult, but is not injected
        into the embedding text.

        This keeps retrieval representation separate from
        evidence/display metadata.
        """

        return chunk.text

    # =========================================================
    # Excerpt
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

        if len(collapsed) <= max_length:

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