from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections import deque
from collections.abc import Iterable

from repobrain.models.graph import (
    GraphEdge,
    GraphNeighbor,
    GraphNode,
    GraphStats,
    GraphTraversalStep,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    RelationshipType,
)


class RepositoryKnowledgeGraph:
    """
    Deterministic in-memory repository knowledge graph.

    Phase 4 intentionally builds only internal symbol-to-symbol
    edges whose source and target can both be resolved to known
    repository symbols.

    Unresolved/external relationships are retained separately
    for diagnostics but are not inserted into graph adjacency
    structures.

    This prevents RepoBrain from pretending that an unresolved
    relationship points to a proven internal symbol.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[CodeSymbol],
        relationships: Iterable[
            CodeRelationship
        ],
    ) -> None:

        self._symbols: dict[
            str,
            CodeSymbol,
        ] = {}

        self._nodes: dict[
            str,
            GraphNode,
        ] = {}

        self._edges: dict[
            str,
            GraphEdge,
        ] = {}

        self._outgoing: dict[
            str,
            list[GraphEdge],
        ] = defaultdict(list)

        self._incoming: dict[
            str,
            list[GraphEdge],
        ] = defaultdict(list)

        self._unresolved_relationships: list[
            CodeRelationship
        ] = []

        self._build(
            symbols=symbols,
            relationships=relationships,
        )

    # =========================================================
    # Construction
    # =========================================================

    def _build(
        self,
        *,
        symbols: Iterable[CodeSymbol],
        relationships: Iterable[
            CodeRelationship
        ],
    ) -> None:

        for symbol in symbols:

            if (
                symbol.symbol_id
                in self._symbols
            ):
                raise ValueError(
                    "Duplicate symbol_id in knowledge graph: "
                    f"{symbol.symbol_id}"
                )

            self._symbols[
                symbol.symbol_id
            ] = symbol

            self._nodes[
                symbol.symbol_id
            ] = self._node_from_symbol(
                symbol
            )

        for relationship in relationships:

            source_symbol_id = (
                relationship.source_symbol_id
            )

            target_symbol_id = (
                relationship.target_symbol_id
            )

            # -------------------------------------------------
            # Only proven internal symbol-to-symbol edges enter
            # the graph.
            # -------------------------------------------------

            if (
                source_symbol_id is None
                or target_symbol_id is None
            ):
                self._unresolved_relationships.append(
                    relationship
                )
                continue

            if (
                source_symbol_id
                not in self._nodes
                or target_symbol_id
                not in self._nodes
            ):
                self._unresolved_relationships.append(
                    relationship
                )
                continue

            if (
                relationship.relationship_id
                in self._edges
            ):
                raise ValueError(
                    "Duplicate relationship_id in "
                    "knowledge graph: "
                    f"{relationship.relationship_id}"
                )

            edge = self._edge_from_relationship(
                relationship
            )

            self._edges[
                edge.relationship_id
            ] = edge

            self._outgoing[
                source_symbol_id
            ].append(
                edge
            )

            self._incoming[
                target_symbol_id
            ].append(
                edge
            )

        self._sort_adjacency()

    def _sort_adjacency(
        self,
    ) -> None:

        def edge_key(
            edge: GraphEdge,
        ) -> tuple[
            str,
            str,
            str,
        ]:

            return (
                edge.relationship_type.value,
                edge.target_qualified_name.casefold(),
                edge.relationship_id,
            )

        for edges in (
            self._outgoing.values()
        ):
            edges.sort(
                key=edge_key
            )

        for edges in (
            self._incoming.values()
        ):

            edges.sort(
                key=lambda edge: (
                    edge.relationship_type.value,
                    edge.source_qualified_name.casefold(),
                    edge.relationship_id,
                )
            )

    # =========================================================
    # Conversion
    # =========================================================

    @staticmethod
    def _node_from_symbol(
        symbol: CodeSymbol,
    ) -> GraphNode:

        return GraphNode(
            symbol_id=(
                symbol.symbol_id
            ),

            file_id=(
                symbol.file_id
            ),

            repository_id=(
                symbol.repository_id
            ),

            name=(
                symbol.name
            ),

            qualified_name=(
                symbol.qualified_name
            ),

            fully_qualified_name=(
                symbol.fully_qualified_name
            ),

            symbol_type=(
                symbol.symbol_type
            ),

            module=(
                symbol.module
            ),

            start_line=(
                symbol.start_line
            ),

            end_line=(
                symbol.end_line
            ),
        )

    @staticmethod
    def _edge_from_relationship(
        relationship: CodeRelationship,
    ) -> GraphEdge:

        if (
            relationship.source_symbol_id
            is None
        ):
            raise ValueError(
                "Graph edge requires source_symbol_id."
            )

        if (
            relationship.target_symbol_id
            is None
        ):
            raise ValueError(
                "Graph edge requires target_symbol_id."
            )

        return GraphEdge(
            relationship_id=(
                relationship.relationship_id
            ),

            source_symbol_id=(
                relationship.source_symbol_id
            ),

            target_symbol_id=(
                relationship.target_symbol_id
            ),

            relationship_type=(
                relationship.relationship_type
            ),

            source_qualified_name=(
                relationship.source_qualified_name
            ),

            target_qualified_name=(
                relationship.target_qualified_name
            ),

            file_id=(
                relationship.file_id
            ),

            line_number=(
                relationship.line_number
            ),

            resolution=(
                relationship.resolution
            ),
        )

    # =========================================================
    # Basic lookup
    # =========================================================

    def get_node(
        self,
        symbol_id: str,
    ) -> GraphNode | None:

        return self._nodes.get(
            symbol_id
        )

    def get_symbol(
        self,
        symbol_id: str,
    ) -> CodeSymbol | None:

        return self._symbols.get(
            symbol_id
        )

    def get_edge(
        self,
        relationship_id: str,
    ) -> GraphEdge | None:

        return self._edges.get(
            relationship_id
        )

    # =========================================================
    # Direct adjacency
    # =========================================================

    def outgoing_edges(
        self,
        symbol_id: str,
        *,
        relationship_type: (
            RelationshipType
            | None
        ) = None,
    ) -> list[GraphEdge]:

        edges = list(
            self._outgoing.get(
                symbol_id,
                [],
            )
        )

        if relationship_type is None:
            return edges

        return [
            edge
            for edge in edges
            if (
                edge.relationship_type
                == relationship_type
            )
        ]

    def incoming_edges(
        self,
        symbol_id: str,
        *,
        relationship_type: (
            RelationshipType
            | None
        ) = None,
    ) -> list[GraphEdge]:

        edges = list(
            self._incoming.get(
                symbol_id,
                [],
            )
        )

        if relationship_type is None:
            return edges

        return [
            edge
            for edge in edges
            if (
                edge.relationship_type
                == relationship_type
            )
        ]

    # =========================================================
    # Neighbors
    # =========================================================

    def neighbors(
        self,
        symbol_id: str,
        *,
        relationship_type: (
            RelationshipType
            | None
        ) = None,
        direction: str = "both",
    ) -> list[GraphNeighbor]:

        normalized_direction = (
            direction
            .strip()
            .casefold()
        )

        if normalized_direction not in {
            "outgoing",
            "incoming",
            "both",
        }:
            raise ValueError(
                "direction must be one of: "
                "'outgoing', 'incoming', 'both'"
            )

        results: list[
            GraphNeighbor
        ] = []

        if normalized_direction in {
            "outgoing",
            "both",
        }:

            for edge in self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    relationship_type
                ),
            ):

                node = self._nodes.get(
                    edge.target_symbol_id
                )

                if node is None:
                    continue

                results.append(
                    GraphNeighbor(
                        symbol=node,
                        relationship=edge,
                        direction="outgoing",
                    )
                )

        if normalized_direction in {
            "incoming",
            "both",
        }:

            for edge in self.incoming_edges(
                symbol_id,
                relationship_type=(
                    relationship_type
                ),
            ):

                node = self._nodes.get(
                    edge.source_symbol_id
                )

                if node is None:
                    continue

                results.append(
                    GraphNeighbor(
                        symbol=node,
                        relationship=edge,
                        direction="incoming",
                    )
                )

        return results

    # =========================================================
    # CALL graph convenience
    # =========================================================

    def callees(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._target_nodes(
            self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.CALLS
                ),
            )
        )

    def callers(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._source_nodes(
            self.incoming_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.CALLS
                ),
            )
        )

    # =========================================================
    # Inheritance graph convenience
    # =========================================================

    def parents(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._target_nodes(
            self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.INHERITS
                ),
            )
        )

    def children(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._source_nodes(
            self.incoming_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.INHERITS
                ),
            )
        )

    # =========================================================
    # Definitions
    # =========================================================

    def defined_symbols(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._target_nodes(
            self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.DEFINES
                ),
            )
        )

    def defined_by(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._source_nodes(
            self.incoming_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.DEFINES
                ),
            )
        )

    # =========================================================
    # Imports
    # =========================================================

    def imports(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._target_nodes(
            self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.IMPORTS
                ),
            )
        )

    def imported_by(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._source_nodes(
            self.incoming_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.IMPORTS
                ),
            )
        )

    # =========================================================
    # Decorators
    # =========================================================

    def decorators(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._target_nodes(
            self.outgoing_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.DECORATED_BY
                ),
            )
        )

    def decorated_symbols(
        self,
        symbol_id: str,
    ) -> list[GraphNode]:

        return self._source_nodes(
            self.incoming_edges(
                symbol_id,
                relationship_type=(
                    RelationshipType.DECORATED_BY
                ),
            )
        )

    # =========================================================
    # Bounded traversal
    # =========================================================

    def traverse(
        self,
        symbol_id: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        relationship_types: (
            set[RelationshipType]
            | None
        ) = None,
    ) -> list[
        GraphTraversalStep
    ]:
        """
        Breadth-first traversal around one symbol.

        The starting node is included at depth 0.

        Each symbol is visited at most once, preventing cycles
        from producing infinite traversal.
        """

        if max_depth < 0:
            raise ValueError(
                "max_depth must be >= 0"
            )

        start = self._nodes.get(
            symbol_id
        )

        if start is None:
            return []

        normalized_direction = (
            direction
            .strip()
            .casefold()
        )

        if normalized_direction not in {
            "outgoing",
            "incoming",
            "both",
        }:
            raise ValueError(
                "direction must be one of: "
                "'outgoing', 'incoming', 'both'"
            )

        results: list[
            GraphTraversalStep
        ] = [
            GraphTraversalStep(
                symbol=start,
                depth=0,
            )
        ]

        visited = {
            symbol_id,
        }

        queue: deque[
            tuple[
                str,
                int,
            ]
        ] = deque(
            [
                (
                    symbol_id,
                    0,
                )
            ]
        )

        while queue:

            (
                current_symbol_id,
                current_depth,
            ) = queue.popleft()

            if (
                current_depth
                >= max_depth
            ):
                continue

            neighbors = self.neighbors(
                current_symbol_id,
                direction=(
                    normalized_direction
                ),
            )

            for neighbor in neighbors:

                edge = (
                    neighbor.relationship
                )

                if (
                    relationship_types
                    is not None
                    and edge.relationship_type
                    not in relationship_types
                ):
                    continue

                neighbor_id = (
                    neighbor.symbol.symbol_id
                )

                if neighbor_id in visited:
                    continue

                visited.add(
                    neighbor_id
                )

                next_depth = (
                    current_depth
                    + 1
                )

                results.append(
                    GraphTraversalStep(
                        symbol=(
                            neighbor.symbol
                        ),

                        depth=(
                            next_depth
                        ),

                        via_relationship=(
                            edge
                        ),

                        direction=(
                            neighbor.direction
                        ),
                    )
                )

                queue.append(
                    (
                        neighbor_id,
                        next_depth,
                    )
                )

        return results

    # =========================================================
    # Diagnostics
    # =========================================================

    def unresolved_relationships(
        self,
    ) -> list[
        CodeRelationship
    ]:

        return list(
            self._unresolved_relationships
        )

    def stats(
        self,
    ) -> GraphStats:

        counts = Counter(
            edge.relationship_type.value
            for edge
            in self._edges.values()
        )

        return GraphStats(
            nodes=len(
                self._nodes
            ),

            edges=len(
                self._edges
            ),

            unresolved_relationships=len(
                self._unresolved_relationships
            ),

            edges_by_type=dict(
                sorted(
                    counts.items()
                )
            ),
        )

    def all_nodes(
        self,
    ) -> list[GraphNode]:

        return sorted(
            self._nodes.values(),
            key=lambda node: (
                node.qualified_name.casefold(),
                node.symbol_id,
            ),
        )

    def all_edges(
        self,
    ) -> list[GraphEdge]:

        return sorted(
            self._edges.values(),
            key=lambda edge: (
                edge.relationship_type.value,
                edge.source_qualified_name.casefold(),
                edge.target_qualified_name.casefold(),
                edge.relationship_id,
            ),
        )

    # =========================================================
    # Internal helpers
    # =========================================================

    def _target_nodes(
        self,
        edges: Iterable[
            GraphEdge
        ],
    ) -> list[GraphNode]:

        nodes: list[
            GraphNode
        ] = []

        for edge in edges:

            node = self._nodes.get(
                edge.target_symbol_id
            )

            if node is not None:
                nodes.append(
                    node
                )

        return nodes

    def _source_nodes(
        self,
        edges: Iterable[
            GraphEdge
        ],
    ) -> list[GraphNode]:

        nodes: list[
            GraphNode
        ] = []

        for edge in edges:

            node = self._nodes.get(
                edge.source_symbol_id
            )

            if node is not None:
                nodes.append(
                    node
                )

        return nodes

    # =========================================================
    # Container helpers
    # =========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self._nodes
        )