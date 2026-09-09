from __future__ import annotations

from typing import Any

from repobrain.evidence import (
    EvidenceAssembler,
)

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.models.agent import (
    AgentGraphFact,
)

from repobrain.models.evidence import (
    EvidenceBundle,
)

from repobrain.models.symbols import (
    RelationshipType,
)


class AgentToolbox:
    """
    Deterministic tool surface exposed to the Phase 7 orchestrator.

    No shell.
    No filesystem mutation.
    No arbitrary Python execution.
    """

    def __init__(
        self,
        *,
        hybrid_engine: Any,
        evidence_assembler: EvidenceAssembler,
        graph: RepositoryKnowledgeGraph,
        symbol_index: Any,
    ) -> None:

        self.hybrid_engine = (
            hybrid_engine
        )

        self.evidence_assembler = (
            evidence_assembler
        )

        self.graph = graph

        self.symbol_index = (
            symbol_index
        )

    # =====================================================================
    # Hybrid retrieval + evidence
    # =====================================================================

    def retrieve_evidence(
        self,
        query: str,
        *,
        retrieval_top_k: int = 20,
    ) -> EvidenceBundle:

        results = (
            self.hybrid_engine.search(
                query,
                top_k=(
                    retrieval_top_k
                ),
            )
        )

        return (
            self.evidence_assembler
            .assemble(
                query=query,
                results=results,
            )
        )

    # =====================================================================
    # Symbol resolution
    # =====================================================================

    def resolve_symbol(
        self,
        reference: str,
    ) -> Any | None:
        """
        Conservative symbol resolution for explicit graph questions.

        Resolution order:

            exact qualified name
            exact fully-qualified name
            exact short name
            unique qualified-name suffix

        Ambiguous matches return None.
        """

        normalized = (
            reference
            .strip()
            .casefold()
        )

        if not normalized:
            return None

        symbols = list(
            self.symbol_index
            .all_symbols()
        )

        exact_qualified = [
            symbol
            for symbol in symbols
            if (
                symbol.qualified_name
                .casefold()
                == normalized
            )
        ]

        if len(
            exact_qualified
        ) == 1:
            return exact_qualified[0]

        exact_fully_qualified = [
            symbol
            for symbol in symbols
            if (
                symbol.fully_qualified_name
                .casefold()
                == normalized
            )
        ]

        if len(
            exact_fully_qualified
        ) == 1:
            return exact_fully_qualified[0]

        exact_name = [
            symbol
            for symbol in symbols
            if (
                symbol.name
                .casefold()
                == normalized
            )
        ]

        if len(
            exact_name
        ) == 1:
            return exact_name[0]

        suffix = [
            symbol
            for symbol in symbols
            if (
                symbol.qualified_name
                .casefold()
                .endswith(
                    f".{normalized}"
                )
            )
        ]

        if len(suffix) == 1:
            return suffix[0]

        return None

    # =====================================================================
    # Callers
    # =====================================================================

    def callers(
        self,
        symbol_id: str,
    ) -> list[
        AgentGraphFact
    ]:

        target = self.graph.get_node(
            symbol_id
        )

        if target is None:
            return []

        facts: list[
            AgentGraphFact
        ] = []

        for caller in (
            self.graph.callers(
                symbol_id
            )
        ):

            facts.append(
                AgentGraphFact(
                    relationship_type=(
                        RelationshipType.CALLS
                    ),
                    source_symbol_id=(
                        caller.symbol_id
                    ),
                    source_qualified_name=(
                        caller.qualified_name
                    ),
                    target_symbol_id=(
                        target.symbol_id
                    ),
                    target_qualified_name=(
                        target.qualified_name
                    ),
                )
            )

        return facts

    # =====================================================================
    # Callees
    # =====================================================================

    def callees(
        self,
        symbol_id: str,
    ) -> list[
        AgentGraphFact
    ]:

        source = self.graph.get_node(
            symbol_id
        )

        if source is None:
            return []

        facts: list[
            AgentGraphFact
        ] = []

        for callee in (
            self.graph.callees(
                symbol_id
            )
        ):

            facts.append(
                AgentGraphFact(
                    relationship_type=(
                        RelationshipType.CALLS
                    ),
                    source_symbol_id=(
                        source.symbol_id
                    ),
                    source_qualified_name=(
                        source.qualified_name
                    ),
                    target_symbol_id=(
                        callee.symbol_id
                    ),
                    target_qualified_name=(
                        callee.qualified_name
                    ),
                )
            )

        return facts