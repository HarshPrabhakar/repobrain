from __future__ import annotations

from pydantic import BaseModel, Field

from repobrain.models.symbols import (
    RelationshipType,
    ResolutionType,
    SymbolType,
)


class GraphNode(BaseModel):
    """
    Lightweight graph representation of one repository symbol.
    """

    symbol_id: str

    file_id: str
    repository_id: str

    name: str
    qualified_name: str
    fully_qualified_name: str

    symbol_type: SymbolType

    module: str

    start_line: int
    end_line: int


class GraphEdge(BaseModel):
    """
    One resolved internal repository graph edge.
    """

    relationship_id: str

    source_symbol_id: str
    target_symbol_id: str

    relationship_type: RelationshipType

    source_qualified_name: str
    target_qualified_name: str

    file_id: str

    line_number: int | None = None

    resolution: ResolutionType


class GraphNeighbor(BaseModel):
    """
    One neighboring symbol plus the relationship connecting it
    to the source symbol.
    """

    symbol: GraphNode

    relationship: GraphEdge

    direction: str


class GraphTraversalStep(BaseModel):
    """
    One symbol reached during bounded graph traversal.
    """

    symbol: GraphNode

    depth: int = Field(
        ge=0,
    )

    via_relationship: GraphEdge | None = None

    direction: str | None = None


class GraphStats(BaseModel):
    """
    Summary information for one repository knowledge graph.
    """

    nodes: int = Field(
        ge=0,
    )

    edges: int = Field(
        ge=0,
    )

    unresolved_relationships: int = Field(
        ge=0,
    )

    edges_by_type: dict[str, int] = Field(
        default_factory=dict,
    )