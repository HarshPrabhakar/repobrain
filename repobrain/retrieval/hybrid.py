from __future__ import annotations

from typing import Any

from repobrain.models.hybrid import (
    HybridSearchResult,
)

from repobrain.retrieval.fusion import (
    ReciprocalRankFusion,
)

from repobrain.retrieval.graph_expansion import (
    GraphEvidenceExpander,
)


DEFAULT_HYBRID_CANDIDATE_SIZE = 30

DEFAULT_GRAPH_SEED_COUNT = 8


class HybridRetrievalEngine:
    """
    RepoBrain Phase 5 unified retrieval engine.

    Retrieval channels:

        Symbol Search
        BM25
        Semantic
        Graph evidence

    Graph expansion is seeded from preliminary retrieval
    results rather than operating as an independent search
    engine.
    """

    def __init__(
        self,
        *,
        symbol_engine: Any,
        lexical_engine: Any,
        semantic_engine: Any,
        graph_expander: (
            GraphEvidenceExpander
            | None
        ) = None,
        fusion: (
            ReciprocalRankFusion
            | None
        ) = None,
    ) -> None:

        self.symbol_engine = (
            symbol_engine
        )

        self.lexical_engine = (
            lexical_engine
        )

        self.semantic_engine = (
            semantic_engine
        )

        self.graph_expander = (
            graph_expander
        )

        self.fusion = (
            fusion
            or ReciprocalRankFusion()
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_size: int = (
            DEFAULT_HYBRID_CANDIDATE_SIZE
        ),
        graph_seed_count: int = (
            DEFAULT_GRAPH_SEED_COUNT
        ),
        graph_depth: int = 1,
    ) -> list[
        HybridSearchResult
    ]:

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            return []

        if top_k <= 0:
            return []

        candidate_size = max(
            candidate_size,
            top_k,
        )

        # =====================================================
        # Independent retrievers
        # =====================================================

        symbol_results = (
            self.symbol_engine.search(
                normalized_query,
                top_k=candidate_size,
            )
        )

        lexical_results = (
            self.lexical_engine.search(
                normalized_query,
                top_k=candidate_size,
            )
        )

        semantic_results = (
            self.semantic_engine.search(
                normalized_query,
                top_k=candidate_size,
            )
        )

        # =====================================================
        # Preliminary fusion
        # =====================================================

        preliminary = (
            self.fusion.fuse(
                symbol_results=(
                    symbol_results
                ),

                lexical_results=(
                    lexical_results
                ),

                semantic_results=(
                    semantic_results
                ),

                top_k=(
                    candidate_size
                ),
            )
        )

        # =====================================================
        # Graph seeds
        # =====================================================

        seed_symbol_ids: list[
            str
        ] = []

        seen_seed_ids: set[
            str
        ] = set()

        for result in preliminary:

            symbol_id = (
                result.symbol_id
            )

            if not symbol_id:
                continue

            if symbol_id in seen_seed_ids:
                continue

            seen_seed_ids.add(
                symbol_id
            )

            seed_symbol_ids.append(
                symbol_id
            )

            if (
                len(seed_symbol_ids)
                >= graph_seed_count
            ):
                break

        # =====================================================
        # Graph expansion
        # =====================================================

        graph_results = []

        if (
            self.graph_expander
            is not None
            and seed_symbol_ids
            and graph_depth > 0
        ):

            graph_results = (
                self.graph_expander.expand(
                    seed_symbol_ids=(
                        seed_symbol_ids
                    ),
                    max_depth=(
                        graph_depth
                    ),
                )
            )

        # =====================================================
        # Final fusion
        # =====================================================

        return self.fusion.fuse(
            symbol_results=(
                symbol_results
            ),

            lexical_results=(
                lexical_results
            ),

            semantic_results=(
                semantic_results
            ),

            graph_results=(
                graph_results
            ),

            top_k=top_k,
        )