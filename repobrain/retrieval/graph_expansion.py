from __future__ import annotations

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.models.hybrid import (
    GraphRetrievalEvidence,
)

from repobrain.models.symbols import (
    RelationshipType,
)


DEFAULT_GRAPH_RELATIONSHIP_TYPES = {
    RelationshipType.CALLS,
    RelationshipType.IMPORTS,
    RelationshipType.INHERITS,
}


class GraphEvidenceExpander:
    """
    Phase 5D bounded graph-assisted retrieval.

    Graph expansion begins only from symbols that have already
    been discovered by another retrieval channel.

    Graph retrieval therefore strengthens existing evidence
    instead of acting as an independent semantic search engine.
    """

    def __init__(
        self,
        graph: RepositoryKnowledgeGraph,
    ) -> None:

        self.graph = graph

    def expand(
        self,
        *,
        seed_symbol_ids: list[str],
        max_depth: int = 1,
        relationship_types: (
            set[RelationshipType]
            | None
        ) = None,
    ) -> list[GraphRetrievalEvidence]:

        if max_depth <= 0:
            return []

        if not seed_symbol_ids:
            return []

        allowed_types = (
            relationship_types
            if relationship_types is not None
            else DEFAULT_GRAPH_RELATIONSHIP_TYPES
        )

        results: list[
            GraphRetrievalEvidence
        ] = []

        seen: set[
            tuple[
                str,
                str,
                RelationshipType,
                int,
            ]
        ] = set()

        for seed_symbol_id in seed_symbol_ids:

            traversal = self.graph.traverse(
                seed_symbol_id,
                max_depth=max_depth,
                direction="both",
                relationship_types=allowed_types,
            )

            for step in traversal:

                if step.depth == 0:
                    continue

                relationship = (
                    step.via_relationship
                )

                if relationship is None:
                    continue

                key = (
                    seed_symbol_id,
                    step.symbol.symbol_id,
                    relationship.relationship_type,
                    step.depth,
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                results.append(
                    GraphRetrievalEvidence(
                        symbol_id=(
                            step.symbol.symbol_id
                        ),

                        qualified_name=(
                            step.symbol.qualified_name
                        ),

                        file_id=(
                            step.symbol.file_id
                        ),

                        relationship_type=(
                            relationship.relationship_type
                        ),

                        hop_distance=(
                            step.depth
                        ),

                        seed_symbol_id=(
                            seed_symbol_id
                        ),

                        direction=(
                            step.direction
                            or "unknown"
                        ),
                    )
                )

        results.sort(
            key=lambda item: (
                item.hop_distance,
                item.relationship_type.value,
                item.qualified_name.casefold(),
                item.symbol_id,
            )
        )

        return results