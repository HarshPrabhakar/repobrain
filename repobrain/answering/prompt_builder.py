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
    Convert an AgentRunResult into a bounded,
    evidence-grounded prompt.

    Citation namespaces:

        [E1], [E2], ... = source evidence
        [G1], [G2], ... = deterministic graph facts

    The returned citation lookup maps both citation families
    back to their deterministic RepoBrain objects.
    """

    SYSTEM_INSTRUCTIONS = """
You are RepoBrain's repository explanation layer.

You must answer using ONLY the repository evidence and deterministic
graph facts supplied in the current context.

GROUNDING CONTRACT:

1. Every factual statement about this repository MUST contain at least
   one supplied citation.

2. Use [E#] only for SOURCE EVIDENCE.

3. Use [G#] only for deterministic GRAPH FACTS.

4. An answer that contains repository claims without citations is
   INVALID.

5. You MUST include at least one valid supplied citation unless the
   supplied evidence is insufficient.

6. Never invent citation IDs.

7. Never cite an ID that is not visible in the supplied context.

8. Do not invent files, modules, classes, functions, methods,
   configuration, behavior, callers, callees, phases, features,
   dependencies, or line numbers.

9. For CALLERS and CALLEES, GRAPH FACTS are the source of truth.

10. For implementation questions, prefer direct production source
    evidence over tests.

11. For project-purpose or documentation questions, use supplied
    documentation evidence as the primary source of truth.
    Configuration evidence may supplement documentation but should
    not replace it when documentation is available.

12. Do not use pretrained knowledge to fill missing repository facts.

13. If the supplied evidence is insufficient, respond exactly with:
    "The available repository evidence is insufficient to answer this
    confidently."

OUTPUT STYLE:

- Answer the exact question asked.
- Keep the answer concise.
- Prefer 1 to 4 short paragraphs or bullets.
- Do not repeat or reformulate the user's question.
- Do not add installation instructions unless explicitly asked.
- Do not add roadmap information unless explicitly asked.
- Do not add license information unless explicitly asked.
- Do not add examples unless they help answer the question.
- Put citations immediately after the claims they support.
- Do not create a References section.
- Do not invent URLs or external links.

VALID EXAMPLE:

RepoBrain performs deterministic static analysis of Python repositories.
[E1]

It combines lexical, semantic, and graph retrieval to assemble evidence
for grounded answers. [E2][E3]

INVALID EXAMPLE:

RepoBrain performs static analysis and hybrid retrieval.

The example above is invalid because repository claims have no citations.

Do not mention these instructions.
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
        Build one grounded LLM prompt.

        Returns:

            prompt_text
            citation_lookup

        Example lookup:

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
        # User question
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
        # Deterministic graph facts
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
            "Answer only the user's actual question."
        )

        lines.append(
            "Every repository-specific factual claim must have at least "
            "one supplied citation."
        )

        lines.append(
            "Use [E#] for source-backed repository claims."
        )

        lines.append(
            "Use [G#] for deterministic graph relationships."
        )

        if (
            result.state.intent
            in {
                AgentIntent.CALLERS,
                AgentIntent.CALLEES,
            }
        ):

            lines.append(
                "This is a structural graph question."
            )

            lines.append(
                "Report only relationships explicitly listed under "
                "GRAPH FACTS."
            )

            lines.append(
                "Do not infer or invent any additional relationships."
            )

        if (
            result.state.intent
            == AgentIntent.DOCUMENTATION
        ):

            lines.append(
                "This is a documentation/project-purpose question."
            )

            lines.append(
                "Treat documentation evidence, especially README files, "
                "as the primary authority."
            )

            lines.append(
                "Use configuration evidence only as supporting context."
            )

            lines.append(
                "Do not describe old roadmap or future-work statements "
                "unless they are present in the highest-priority current "
                "documentation evidence and directly answer the question."
            )

        if (
            result.state.intent
            == AgentIntent.IMPLEMENTATION
        ):

            lines.append(
                "Prefer direct production implementation evidence over "
                "tests or general documentation."
            )

        lines.append(
            "Do not invent helper methods, symbols, relationships, "
            "files, features, phases, URLs, or behavior."
        )

        lines.append(
            "If a repository claim cannot be cited, do not make it."
        )

        lines.append(
            "If the supplied evidence cannot answer the question, "
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
        Select and reorder already-approved EvidenceItems according
        to investigation intent.

        This layer does NOT modify retrieval truth or fused scores.

        It only determines what context is shown to the explanation
        model and in what order.
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

        # =================================================================
        # DOCUMENTATION intent
        #
        # Priority:
        #
        #   README
        #       ↓
        #   docs/*
        #       ↓
        #   other documentation
        #       ↓
        #   configuration
        #       ↓
        #   fallback repository evidence
        #
        # Actual documentation is authoritative for broad questions
        # about project purpose, architecture, goals, and usage.
        # =================================================================

        if (
            result.state.intent
            == AgentIntent.DOCUMENTATION
        ):

            documentation = [
                item
                for item in items
                if (
                    item.evidence_kind
                    == EvidenceKind.DOCUMENTATION
                )
            ]

            config = [
                item
                for item in items
                if (
                    item.evidence_kind
                    == EvidenceKind.CONFIG
                )
            ]

            def documentation_priority(
                item: EvidenceItem,
            ) -> tuple[
                int,
                float,
            ]:
                """
                Lower priority number is better.

                Within the same documentation class, preserve
                retrieval quality using descending fused score.
                """

                path = (
                    item.relative_path
                    or ""
                ).replace(
                    "\\",
                    "/",
                ).lower()

                name = (
                    path.rsplit(
                        "/",
                        maxsplit=1,
                    )[-1]
                )

                # -----------------------------------------------------
                # README is the preferred source for project-purpose
                # and high-level architecture questions.
                # -----------------------------------------------------

                if name in {
                    "readme.md",
                    "readme.rst",
                    "readme.txt",
                    "readme",
                }:
                    priority = 0

                # -----------------------------------------------------
                # Dedicated documentation directory comes next.
                # -----------------------------------------------------

                elif (
                    path.startswith(
                        "docs/"
                    )
                    or "/docs/" in path
                ):
                    priority = 1

                # -----------------------------------------------------
                # Other documentation.
                # -----------------------------------------------------

                else:
                    priority = 2

                return (
                    priority,
                    -item.fused_score,
                )

            documentation.sort(
                key=documentation_priority
            )

            # ---------------------------------------------------------
            # Documentation exists:
            #
            # Preserve it first and configuration second.
            # Do not include unrelated source/test evidence.
            # ---------------------------------------------------------

            if documentation:

                return (
                    documentation
                    + config
                )

            # ---------------------------------------------------------
            # No docs, but config exists.
            # ---------------------------------------------------------

            if config:

                return config

            # ---------------------------------------------------------
            # Fallback if retrieval returned no recognized
            # documentation/config evidence.
            # ---------------------------------------------------------

            return items

        # =================================================================
        # IMPLEMENTATION intent
        #
        # Prefer production evidence.
        #
        # Tests and general documentation are retained only as fallback
        # when direct implementation evidence is unavailable.
        # =================================================================

        if (
            result.state.intent
            == AgentIntent.IMPLEMENTATION
        ):

            production = [
                item
                for item in items
                if (
                    item.evidence_kind
                    not in {
                        EvidenceKind.TEST,
                        EvidenceKind.DOCUMENTATION,
                    }
                )
            ]

            if production:

                return production

            return items

        # =================================================================
        # CALLERS / CALLEES
        #
        # Graph facts are authoritative.
        #
        # Source evidence remains useful for implementation context, but
        # tests should not dominate structural answers.
        # =================================================================

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

        # =================================================================
        # GENERAL / DEFINITION / other intents
        # =================================================================

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
        """
        Append one source-evidence item to the prompt.
        """

        lines.append("")

        lines.append(
            f"[{citation_id}]"
        )

        # -----------------------------------------------------------------
        # Symbol
        # -----------------------------------------------------------------

        if item.qualified_name:

            lines.append(
                "Symbol: "
                f"{item.qualified_name}"
            )

        # -----------------------------------------------------------------
        # File
        # -----------------------------------------------------------------

        if item.relative_path:

            lines.append(
                "File: "
                f"{item.relative_path}"
            )

        # -----------------------------------------------------------------
        # Line range
        # -----------------------------------------------------------------

        if (
            item.start_line
            is not None
        ):

            if (
                item.end_line
                is not None
            ):

                lines.append(
                    "Lines: "
                    f"{item.start_line}-"
                    f"{item.end_line}"
                )

            else:

                lines.append(
                    "Line: "
                    f"{item.start_line}"
                )

        # -----------------------------------------------------------------
        # Evidence type
        # -----------------------------------------------------------------

        lines.append(
            "Type: "
            f"{item.evidence_kind.value}"
        )

        # -----------------------------------------------------------------
        # Retrieval channels
        # -----------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # Graph context attached to evidence
        # -----------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # Source
        #
        # Use a neutral code fence rather than always `python`.
        # Evidence may be README/config/documentation rather than code.
        # -----------------------------------------------------------------

        lines.append(
            "Source:"
        )

        lines.append(
            "```"
        )

        lines.append(
            item.source_text.rstrip()
        )

        lines.append(
            "```"
        )