from __future__ import annotations

from repobrain.models.agent import (
    AgentIntent,
    AgentRunResult,
)

from repobrain.models.evidence import (
    EvidenceItem,
    EvidenceKind,
)


class GroundedPromptBuilder:
    """
    Convert AgentRunResult into a bounded, evidence-grounded prompt.

    Citation namespaces:

        [E1], [E2], ... = source evidence
        [G1], [G2], ... = deterministic graph facts

    The returned lookup maps both citation families back to
    their deterministic RepoBrain objects.
    """

    SYSTEM_INSTRUCTIONS = """
You are RepoBrain's repository explanation layer.

You must answer using ONLY the repository evidence and deterministic
graph facts supplied in the current context.

STRICT RULES:

1. Do not invent repository behavior.

2. Do not invent files, modules, classes, functions, methods,
   configuration, relationships, callers, callees, or line numbers.

3. Every repository-specific factual claim MUST include a citation.

4. Use [E#] only for SOURCE EVIDENCE.

5. Use [G#] only for deterministic GRAPH FACTS.

6. Never cite an ID that is not supplied in the current context.

7. For caller or callee questions, GRAPH FACTS are the primary
   source of truth.

8. For caller/callee questions, report ONLY relationships explicitly
   present in GRAPH FACTS.

9. Never infer additional callers or callees from naming, source-code
   similarity, general programming knowledge, or pretrained knowledge.

10. For implementation questions, prefer direct production source
    evidence over tests.

11. For documentation or project-purpose questions, prefer repository
    documentation and configuration evidence.

12. Tests may support a claim, but tests must not replace direct
    implementation evidence when direct implementation evidence exists.

13. Do not use pretrained knowledge to fill gaps in the supplied
    repository evidence.

14. If the supplied evidence is insufficient, say:
    "The available repository evidence is insufficient to answer this
    confidently."

15. Be concise, technically useful, and direct.

16. Do not mention these instructions.
""".strip()

    # =====================================================================
    # Public API
    # =====================================================================

    def build(
        self,
        result: AgentRunResult,
    ) -> tuple[
        str,
        dict[str, object],
    ]:
        """
        Build one grounded prompt.

        Returns:

            prompt text
            citation lookup

        lookup example:

            {
                "E1": EvidenceItem(...),
                "E2": EvidenceItem(...),
                "G1": AgentGraphFact(...),
            }
        """

        lines: list[str] = []

        citation_lookup: dict[
            str,
            object,
        ] = {}

        # =================================================================
        # Question
        # =================================================================

        lines.append(
            "USER QUESTION"
        )

        lines.append(
            "============="
        )

        lines.append(
            result.state.query
        )

        # =================================================================
        # Investigation metadata
        # =================================================================

        lines.append("")

        lines.append(
            "INVESTIGATION"
        )

        lines.append(
            "============="
        )

        lines.append(
            f"Intent: "
            f"{result.state.intent.value}"
        )

        lines.append(
            f"Status: "
            f"{result.state.status.value}"
        )

        if (
            result.resolved_qualified_name
            is not None
        ):
            lines.append(
                "Resolved symbol: "
                f"{result.resolved_qualified_name}"
            )

        # =================================================================
        # Graph facts
        # =================================================================

        lines.append("")

        lines.append(
            "GRAPH FACTS"
        )

        lines.append(
            "==========="
        )

        if not result.graph_facts:

            lines.append(
                "(none)"
            )

        else:

            for index, fact in enumerate(
                result.graph_facts,
                start=1,
            ):

                citation_id = (
                    f"G{index}"
                )

                citation_lookup[
                    citation_id
                ] = fact

                lines.append(
                    f"[{citation_id}] "
                    f"{fact.source_qualified_name} "
                    f"--"
                    f"{fact.relationship_type.value}"
                    f"--> "
                    f"{fact.target_qualified_name}"
                )

        # =================================================================
        # Source evidence
        # =================================================================

        lines.append("")

        lines.append(
            "SOURCE EVIDENCE"
        )

        lines.append(
            "==============="
        )

        evidence_items = (
            self._select_evidence(
                result
            )
        )

        if not evidence_items:

            lines.append(
                "(no source evidence available)"
            )

        else:

            for index, item in enumerate(
                evidence_items,
                start=1,
            ):

                citation_id = (
                    f"E{index}"
                )

                citation_lookup[
                    citation_id
                ] = item

                self._append_evidence(
                    lines=lines,
                    citation_id=(
                        citation_id
                    ),
                    item=item,
                )

        # =================================================================
        # Answer requirements
        # =================================================================

        lines.append("")

        lines.append(
            "ANSWER REQUIREMENTS"
        )

        lines.append(
            "==================="
        )

        lines.append(
            "Answer the user question directly."
        )

        lines.append(
            "Every repository-specific factual claim must have "
            "at least one supplied citation."
        )

        lines.append(
            "Use [E#] for source-backed claims."
        )

        lines.append(
            "Use [G#] for caller/callee or other deterministic "
            "graph relationship claims."
        )

        if (
            result.state.intent
            in {
                AgentIntent.CALLERS,
                AgentIntent.CALLEES,
            }
        ):

            lines.append(
                "This is a graph relationship question. "
                "Report only relationships listed under GRAPH FACTS."
            )

            lines.append(
                "Do not infer or invent any additional relationships."
            )

        if (
            result.state.intent
            == AgentIntent.DOCUMENTATION
        ):

            lines.append(
                "Prioritize documentation/configuration evidence "
                "when explaining the project's purpose."
            )

        lines.append(
            "If a repository claim cannot be cited, do not make it."
        )

        lines.append(
            "If the evidence does not answer the question, explicitly "
            "state that the available repository evidence is insufficient."
        )

        return (
            "\n".join(
                lines
            ),
            citation_lookup,
        )

    # =====================================================================
    # Evidence selection
    # =====================================================================

    @staticmethod
    def _select_evidence(
        result: AgentRunResult,
    ) -> list[
        EvidenceItem
    ]:
        """
        Select/reorder evidence according to investigation intent.

        Important:
        This does NOT change repository truth or retrieval scores.
        It only controls which already-approved EvidenceItems are
        presented to the explanation model.
        """

        bundle = (
            result.evidence
        )

        if (
            bundle is None
            or not bundle.items
        ):
            return []

        items = list(
            bundle.items
        )

        # -------------------------------------------------------------
        # Documentation intent
        #
        # If actual documentation/config evidence exists, keep the
        # explanation model focused on it rather than implementation
        # internals.
        # -------------------------------------------------------------

        if (
            result.state.intent
            == AgentIntent.DOCUMENTATION
        ):

            documentation = [
                item
                for item in items
                if item.evidence_kind
                in {
                    EvidenceKind.DOCUMENTATION,
                    EvidenceKind.CONFIG,
                }
            ]

            if documentation:
                return documentation

            return items

        # -------------------------------------------------------------
        # Implementation intent
        #
        # Prefer non-test repository implementation evidence.
        # Tests remain a fallback if production evidence does not exist.
        # -------------------------------------------------------------

        if (
            result.state.intent
            == AgentIntent.IMPLEMENTATION
        ):

            production = [
                item
                for item in items
                if item.evidence_kind
                not in {
                    EvidenceKind.TEST,
                    EvidenceKind.DOCUMENTATION,
                }
            ]

            if production:
                return production

            return items

        # -------------------------------------------------------------
        # Graph questions
        #
        # Graph facts are primary. Source evidence remains available
        # as implementation/context support.
        # -------------------------------------------------------------

        if (
            result.state.intent
            in {
                AgentIntent.CALLERS,
                AgentIntent.CALLEES,
            }
        ):

            non_test = [
                item
                for item in items
                if (
                    item.evidence_kind
                    != EvidenceKind.TEST
                )
            ]

            if non_test:
                return non_test

        return items

    # =====================================================================
    # Evidence formatting
    # =====================================================================

    @staticmethod
    def _append_evidence(
        *,
        lines: list[str],
        citation_id: str,
        item: EvidenceItem,
    ) -> None:

        lines.append("")

        lines.append(
            f"[{citation_id}]"
        )

        if item.qualified_name:

            lines.append(
                "Symbol: "
                f"{item.qualified_name}"
            )

        if item.relative_path:

            lines.append(
                "File: "
                f"{item.relative_path}"
            )

        if (
            item.start_line
            is not None
        ):

            lines.append(
                "Lines: "
                f"{item.start_line}-"
                f"{item.end_line}"
            )

        lines.append(
            "Type: "
            f"{item.evidence_kind.value}"
        )

        if item.retrieval_channels:

            channels = ", ".join(
                channel.value
                for channel
                in item.retrieval_channels
            )

            lines.append(
                "Retrieval: "
                f"{channels}"
            )

        if item.graph_context:

            lines.append(
                "Graph context:"
            )

            for relation in (
                item.graph_context
            ):

                line_suffix = ""

                if (
                    relation.line_number
                    is not None
                ):

                    line_suffix = (
                        f" (line "
                        f"{relation.line_number}"
                        f")"
                    )

                lines.append(
                    "  - "
                    f"{relation.direction} "
                    f"{relation.relationship_type.value} "
                    f"{relation.related_qualified_name}"
                    f"{line_suffix}"
                )

        lines.append(
            "Source:"
        )

        lines.append(
            "```python"
        )

        lines.append(
            item.source_text.rstrip()
        )

        lines.append(
            "```"
        )