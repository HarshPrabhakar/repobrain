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

from repobrain.retrieval.semantic_ranking import (
    SemanticRankingPolicy,
)


DEFAULT_CANDIDATE_POOL_SIZE = 50

DEFAULT_CANDIDATE_MULTIPLIER = 5


class SemanticSearchEngine:
    """
    RepoBrain semantic repository search.

    Phase 3C:
        FAISS semantic retrieval.

    Phase 3C.2:
        deterministic AST-aware semantic documents.

    Phase 3C.3:
        deterministic query-intent and source-aware reranking.

    Important:
        Raw semantic similarity remains available as result.score.

        Phase 3C.3 adds result.ranking_score rather than
        overwriting vector similarity.
    """

    def __init__(
        self,
        chunks: list[CodeChunk],
        embedding_provider: EmbeddingProvider,
        *,
        symbols: Iterable[
            CodeSymbol
        ] | None = None,
        relationships: Iterable[
            CodeRelationship
        ] | None = None,
        ranking_policy: (
            SemanticRankingPolicy
            | None
        ) = None,
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

        self._ranking_policy = (
            ranking_policy
            or SemanticRankingPolicy()
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
        rerank: bool = True,
        candidate_pool_size: int | None = None,
    ) -> list[
        SemanticSearchResult
    ]:

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

        # -----------------------------------------------------
        # Candidate pool
        # -----------------------------------------------------
        #
        # Reranking cannot promote a result that FAISS never
        # returned.
        #
        # Example:
        # _calculate_sha256 was previously semantic rank #18.
        #
        # Asking FAISS for only top 10 would make promotion
        # impossible.
        # -----------------------------------------------------

        if candidate_pool_size is None:

            candidate_pool_size = max(
                DEFAULT_CANDIDATE_POOL_SIZE,
                (
                    top_k
                    * DEFAULT_CANDIDATE_MULTIPLIER
                ),
            )

        candidate_pool_size = max(
            top_k,
            candidate_pool_size,
        )

        if language is None:

            search_k = min(
                candidate_pool_size,
                len(self.chunks),
            )

        else:

            # Retrieve the complete index before filtering.
            # This guarantees enough language-specific
            # candidates remain after filtering.
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

        candidates: list[
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

            if (
                document_index
                >= len(self.chunks)
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

            semantic_score = float(
                score
            )

            semantic_score = max(
                -1.0,
                min(
                    1.0,
                    semantic_score,
                ),
            )

            if (
                min_score is not None
                and semantic_score
                < min_score
            ):
                continue

            candidates.append(
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
                        semantic_score
                    ),

                    excerpt=self._excerpt(
                        chunk.text
                    ),
                )
            )

        # -----------------------------------------------------
        # Phase 3C.3
        # -----------------------------------------------------

        if rerank:

            return (
                self._ranking_policy
                .rerank(
                    query=(
                        normalized_query
                    ),
                    results=(
                        candidates
                    ),
                    top_k=top_k,
                )
            )

        return candidates[
            :top_k
        ]

    # =========================================================
    # Semantic representation
    # =========================================================

    def _embedding_text(
        self,
        chunk: CodeChunk,
    ) -> str:

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