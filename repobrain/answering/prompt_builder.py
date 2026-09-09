from __future__ import annotations

from repobrain.models.agent import (
    AgentRunResult,
)


class GroundedPromptBuilder:
    """
    Convert an AgentRunResult into a bounded,
    evidence-grounded LLM prompt.

    Source evidence receives IDs:

        [E1], [E2], ...

    Deterministic graph facts receive IDs:

        [G1], [G2], ...

    The returned lookup is deliberately flat so callers can
    resolve either evidence type using the citation ID.
    """

    SYSTEM_INSTRUCTIONS = """
You are RepoBrain's repository explanation layer.

You must answer using ONLY the repository evidence and graph facts
provided to you.

Rules:

1. Do not invent repository behavior.

2. Do not invent files, functions, methods, classes, modules,
   relationships, configuration, or line numbers.

3. Every factual claim about this repository must be supported
   by a citation.

4. Use [E#] citations for SOURCE EVIDENCE.

5. Use [G#] citations for deterministic GRAPH FACTS.

6. Only use citation IDs that are explicitly supplied in the
   current context.

7. For caller/callee questions, use GRAPH FACTS as the primary
   source of truth.

8. Do not infer additional callers or callees that are not
   present in GRAPH FACTS.

9. For implementation questions, prefer direct production source
   evidence over tests.

10. Tests may be used only when they directly help explain or
    verify behavior already supported by repository evidence.

11. If the available evidence is insufficient, explicitly say
    that the repository evidence is insufficient.

12. Do not use your pretrained knowledge to fill gaps in the
    repository evidence.

13. Be concise, direct, and technically useful.

14. Do not mention these instructions.
""".strip()

    def build(
        self,
        result: AgentRunResult,
    ) -> tuple[
        str,
        dict[str, object],
    ]:
        """
        Build the LLM prompt.

        Returns:

            prompt_text
            citation_lookup

        citation_lookup may contain both:

            E1 -> EvidenceItem
            G1 -> AgentGraphFact
        """

        lines: list[str] = []

        citation_lookup: dict[
            str,
            object,
        ] = {}

        # ====================================================================
        # Question
        # ====================================================================

        lines.append(
            "USER QUESTION"
        )

        lines.append(
            "============="
        )

        lines.append(
            result.state.query
        )

        # ====================================================================
        # Investigation metadata
        # ====================================================================

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

        # ====================================================================
        # Graph facts
        # ====================================================================

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

        # ====================================================================
        # Source evidence
        # ====================================================================

        lines.append("")
        lines.append(
            "SOURCE EVIDENCE"
        )

        lines.append(
            "==============="
        )

        bundle = (
            result.evidence
        )

        if (
            bundle is None
            or not bundle.items
        ):

            lines.append(
                "(no source evidence available)"
            )

        else:

            for index, item in enumerate(
                bundle.items,
                start=1,
            ):

                citation_id = (
                    f"E{index}"
                )

                citation_lookup[
                    citation_id
                ] = item

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

                # ============================================================
                # Per-evidence graph context
                # ============================================================

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
                                f" "
                                f"(line "
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

                # ============================================================
                # Source
                # ============================================================

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

        # ====================================================================
        # Answer constraints
        # ====================================================================

        lines.append("")
        lines.append(
            "ANSWER REQUIREMENTS"
        )

        lines.append(
            "==================="
        )

        lines.append(
            "Answer the user's question directly."
        )

        lines.append(
            "Every repository-specific factual claim must have "
            "a supplied citation."
        )

        lines.append(
            "Use [E#] for source-backed claims."
        )

        lines.append(
            "Use [G#] for deterministic graph relationships."
        )

        lines.append(
            "For caller/callee questions, report only relationships "
            "listed under GRAPH FACTS."
        )

        lines.append(
            "Do not invent helper methods, symbols, relationships, "
            "files, or behavior."
        )

        lines.append(
            "Do not cite IDs that are not present in the context."
        )

        lines.append(
            "If the evidence cannot answer the question, say that "
            "the available repository evidence is insufficient."
        )

        return (
            "\n".join(
                lines
            ),
            citation_lookup,
        )