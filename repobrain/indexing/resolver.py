from __future__ import annotations

from dataclasses import dataclass

from repobrain.indexing.symbol_index import (
    SymbolIndex,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    PythonRepositoryAnalysis,
    ResolutionType,
)


@dataclass(slots=True)
class ResolutionSummary:
    """
    Statistics describing one repository resolution pass.
    """

    total_relationships: int

    already_linked: int

    newly_resolved: int

    unresolved_local_or_external: int


class PythonSymbolResolver:
    """
    Conservative repository-wide symbol resolver.

    Phase 2 extracted textual references such as:

        self.scan
        helper
        repobrain.config.ScannerConfig

    Phase 2.5 attempts to connect those references to actual CodeSymbol
    objects discovered elsewhere in the repository.

    Resolution is performed only when a unique target can be proven.
    """

    def resolve_repository(
        self,
        analysis: PythonRepositoryAnalysis,
    ) -> tuple[
        PythonRepositoryAnalysis,
        ResolutionSummary,
    ]:

        index = SymbolIndex(
            analysis.symbols
        )

        resolved_relationships: list[
            CodeRelationship
        ] = []

        already_linked = 0
        newly_resolved = 0

        for relationship in analysis.relationships:

            if relationship.target_symbol_id:

                already_linked += 1

                resolved_relationships.append(
                    relationship
                )

                continue

            resolved = self.resolve_relationship(
                relationship=relationship,
                index=index,
            )

            if (
                resolved.target_symbol_id
                is not None
            ):
                newly_resolved += 1

            resolved_relationships.append(
                resolved
            )

        unresolved_count = sum(
            1
            for relationship
            in resolved_relationships
            if relationship.target_symbol_id
            is None
        )

        result = analysis.model_copy(
            update={
                "relationships":
                    resolved_relationships
            }
        )

        summary = ResolutionSummary(
            total_relationships=len(
                resolved_relationships
            ),

            already_linked=already_linked,

            newly_resolved=newly_resolved,

            unresolved_local_or_external=(
                unresolved_count
            ),
        )

        return result, summary

    # ---------------------------------------------------------
    # One relationship
    # ---------------------------------------------------------

    def resolve_relationship(
        self,
        relationship: CodeRelationship,
        index: SymbolIndex,
    ) -> CodeRelationship:
        """
        Resolve one relationship to a repository-local target symbol.

        Candidate strategies are deliberately ordered from most specific
        to least specific.
        """

        source = index.get_by_id(
            relationship.source_symbol_id
        )

        if source is None:
            return relationship

        candidates = self._candidate_names(
            relationship=relationship,
            source=source,
            index=index,
        )

        for (
            candidate,
            reason,
        ) in candidates:

            target = (
                index.find_unique_by_qualified_name(
                    candidate
                )
            )

            if target is None:
                continue

            return self._link_relationship(
                relationship=relationship,
                target=target,
                reason=reason,
            )

        return relationship

    # ---------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------

    def _candidate_names(
        self,
        relationship: CodeRelationship,
        source: CodeSymbol,
        index: SymbolIndex,
    ) -> list[
        tuple[str, str]
    ]:

        raw_target = (
            relationship.target_qualified_name
        )

        candidates: list[
            tuple[str, str]
        ] = []

        seen: set[str] = set()

        def add(
            value: str,
            reason: str,
        ) -> None:

            if not value:
                return

            if value in seen:
                return

            seen.add(value)

            candidates.append(
                (
                    value,
                    reason,
                )
            )

        # -----------------------------------------------------
        # Strategy 1:
        # target already looks repository-qualified
        # -----------------------------------------------------

        add(
            raw_target,
            "direct-qualified-match",
        )

        # -----------------------------------------------------
        # Strategy 2:
        # self.method() / cls.method()
        # -----------------------------------------------------

        if (
            raw_target.startswith("self.")
            or raw_target.startswith("cls.")
        ):

            containing_class = (
                index.find_containing_class(
                    source
                )
            )

            if containing_class is not None:

                _, member_path = (
                    raw_target.split(
                        ".",
                        1,
                    )
                )

                add(
                    (
                        f"{containing_class.qualified_name}."
                        f"{member_path}"
                    ),
                    "containing-class-member",
                )

        # -----------------------------------------------------
        # Strategy 3:
        # same module
        #
        # helper()
        # ClassName()
        # ClassName.method()
        # -----------------------------------------------------

        if source.module:

            add(
                f"{source.module}.{raw_target}",
                "same-module",
            )

        return candidates

    # ---------------------------------------------------------
    # Relationship update
    # ---------------------------------------------------------

    @staticmethod
    def _link_relationship(
        relationship: CodeRelationship,
        target: CodeSymbol,
        reason: str,
    ) -> CodeRelationship:

        metadata = dict(
            relationship.metadata
        )

        metadata.setdefault(
            "original_target",
            relationship.target_qualified_name,
        )

        metadata[
            "resolution_reason"
        ] = reason

        metadata[
            "phase"
        ] = "2.5"

        return relationship.model_copy(
            update={
                "target_symbol_id":
                    target.symbol_id,

                "target_qualified_name":
                    target.qualified_name,

                "resolution":
                    ResolutionType.EXACT,

                "metadata":
                    metadata,
            }
        )