from __future__ import annotations

import re

from repobrain.answering.prompt_builder import (
    GroundedPromptBuilder,
)

from repobrain.llm.base import (
    LLMProvider,
)

from repobrain.models.agent import (
    AgentGraphFact,
    AgentIntent,
    AgentRunResult,
)

from repobrain.models.answer import (
    AnswerCitation,
    AnswerCitationKind,
    GroundedAnswer,
)

from repobrain.models.evidence import (
    EvidenceItem,
)


_CITATION_PATTERN = re.compile(
    r"\[(E|G)(\d+)\]"
)


class GroundedAnswerGenerator:
    """
    Generate repository answers while preserving RepoBrain's
    deterministic grounding boundary.

    Structural graph questions:
        CALLERS
        CALLEES

    are rendered directly from deterministic graph facts.

    Other intents use the configured LLM provider.

    If an LLM answer contains repository context but no valid
    citations, RepoBrain performs exactly ONE citation-repair retry.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        prompt_builder: GroundedPromptBuilder | None = None,
    ) -> None:

        self.provider = provider

        self.prompt_builder = (
            prompt_builder
            or GroundedPromptBuilder()
        )

    # =====================================================================
    # Public API
    # =====================================================================

    def generate(
        self,
        result: AgentRunResult,
    ) -> GroundedAnswer:

        # -------------------------------------------------------------
        # Deterministic graph questions
        # -------------------------------------------------------------

        if (
            result.state.intent
            in {
                AgentIntent.CALLERS,
                AgentIntent.CALLEES,
            }
            and result.graph_facts
        ):
            return (
                self._generate_graph_answer(
                    result
                )
            )

        # -------------------------------------------------------------
        # Normal LLM-backed answer
        # -------------------------------------------------------------

        return (
            self._generate_llm_answer(
                result
            )
        )

    # =====================================================================
    # Deterministic graph answer
    # =====================================================================

    def _generate_graph_answer(
        self,
        result: AgentRunResult,
    ) -> GroundedAnswer:

        facts = list(
            result.graph_facts
        )

        citations: list[
            AnswerCitation
        ] = []

        answer_lines: list[str] = []

        # -------------------------------------------------------------
        # CALLERS
        # -------------------------------------------------------------

        if (
            result.state.intent
            == AgentIntent.CALLERS
        ):

            target_name = (
                result.resolved_qualified_name
            )

            if (
                target_name is None
                and facts
            ):
                target_name = (
                    facts[0]
                    .target_qualified_name
                )

            if len(facts) == 1:

                fact = facts[0]

                answer_lines.append(
                    f"`{fact.target_qualified_name}` "
                    f"is called by "
                    f"`{fact.source_qualified_name}`. "
                    f"[G1]"
                )

            else:

                answer_lines.append(
                    f"`{target_name}` is called by:"
                )

                for index, fact in enumerate(
                    facts,
                    start=1,
                ):

                    answer_lines.append(
                        f"{index}. "
                        f"`{fact.source_qualified_name}` "
                        f"[G{index}]"
                    )

        # -------------------------------------------------------------
        # CALLEES
        # -------------------------------------------------------------

        elif (
            result.state.intent
            == AgentIntent.CALLEES
        ):

            source_name = (
                result.resolved_qualified_name
            )

            if (
                source_name is None
                and facts
            ):
                source_name = (
                    facts[0]
                    .source_qualified_name
                )

            answer_lines.append(
                f"`{source_name}` directly calls:"
            )

            for index, fact in enumerate(
                facts,
                start=1,
            ):

                answer_lines.append(
                    f"{index}. "
                    f"`{fact.target_qualified_name}` "
                    f"[G{index}]"
                )

        # -------------------------------------------------------------
        # Graph citations
        # -------------------------------------------------------------

        for index, fact in enumerate(
            facts,
            start=1,
        ):

            citations.append(
                self._graph_citation(
                    citation_id=(
                        f"G{index}"
                    ),
                    fact=fact,
                )
            )

        evidence_count = (
            len(
                result.evidence.items
            )
            if (
                result.evidence
                is not None
            )
            else 0
        )

        return GroundedAnswer(
            query=result.state.query,

            answer_text="\n".join(
                answer_lines
            ),

            model_name=(
                "deterministic-graph"
            ),

            citations=citations,

            invalid_citation_ids=[],

            evidence_items_available=(
                evidence_count
            ),

            graph_facts_available=(
                len(facts)
            ),

            grounded=bool(
                citations
            ),
        )

    # =====================================================================
    # LLM-backed answer
    # =====================================================================

    def _generate_llm_answer(
        self,
        result: AgentRunResult,
    ) -> GroundedAnswer:

        (
            prompt,
            citation_lookup,
        ) = (
            self.prompt_builder.build(
                result
            )
        )

        # -------------------------------------------------------------
        # First generation attempt
        # -------------------------------------------------------------

        answer_text = (
            self.provider.generate(
                instructions=(
                    self.prompt_builder
                    .SYSTEM_INSTRUCTIONS
                ),
                input_text=prompt,
            )
        )

        (
            valid_citations,
            invalid_citations,
        ) = (
            self._validate_citations(
                answer_text=answer_text,
                citation_lookup=(
                    citation_lookup
                ),
            )
        )

        bundle = (
            result.evidence
        )

        evidence_count = (
            len(
                bundle.items
            )
            if bundle
            is not None
            else 0
        )

        graph_count = len(
            result.graph_facts
        )

        has_repository_context = (
            evidence_count > 0
            or graph_count > 0
        )

        # -------------------------------------------------------------
        # One-pass citation repair
        #
        # Retry only when:
        #
        # - repository context exists
        # - no valid citations were produced
        #
        # This covers the common case where the model answered from
        # evidence but simply forgot citation syntax.
        # -------------------------------------------------------------

        if (
            has_repository_context
            and (
                not valid_citations
                or invalid_citations
            )
        ):

            repair_prompt = (
                self._build_citation_repair_prompt(
                    original_prompt=prompt,
                    original_answer=answer_text,
                )
            )

            repaired_answer_text = (
                self.provider.generate(
                    instructions=(
                        self.prompt_builder
                        .SYSTEM_INSTRUCTIONS
                    ),
                    input_text=(
                        repair_prompt
                    ),
                )
            )

            (
                repaired_valid_citations,
                repaired_invalid_citations,
            ) = (
                self._validate_citations(
                    answer_text=(
                        repaired_answer_text
                    ),
                    citation_lookup=(
                        citation_lookup
                    ),
                )
            )

            # ---------------------------------------------------------
            # Accept the repair only if it improves grounding.
            #
            # If the retry still has no valid citations, we preserve
            # the repaired text but Grounded will remain False.
            # ---------------------------------------------------------

            answer_text = (
                repaired_answer_text
            )

            valid_citations = (
                repaired_valid_citations
            )

            invalid_citations = (
                repaired_invalid_citations
            )

        grounded = (
            has_repository_context
            and bool(
                valid_citations
            )
            and not invalid_citations
        )

        return GroundedAnswer(
            query=result.state.query,

            answer_text=answer_text,

            model_name=(
                self.provider.model_name
            ),

            citations=(
                valid_citations
            ),

            invalid_citation_ids=(
                invalid_citations
            ),

            evidence_items_available=(
                evidence_count
            ),

            graph_facts_available=(
                graph_count
            ),

            grounded=grounded,
        )

    # =====================================================================
    # Citation repair prompt
    # =====================================================================

    @staticmethod
    def _build_citation_repair_prompt(
        *,
        original_prompt: str,
        original_answer: str,
    ) -> str:
        """
        Perform exactly one grounding/citation repair.

        The model must rewrite the answer using only supplied
        repository evidence and exact citation tokens.
        """

        return (
            f"{original_prompt}\n\n"
            "GROUNDING VALIDATION FAILURE\n"
            "============================\n\n"
            "RepoBrain rejected your previous answer because it either "
            "contained no valid citations or contained citation IDs that "
            "were not supplied.\n\n"

            "REWRITE CONTRACT\n"
            "================\n"
            "1. Return ONLY the rewritten answer.\n"
            "2. Answer only the user's actual question.\n"
            "3. Keep the answer concise: at most 4 short paragraphs "
            "or bullets.\n"
            "4. Every repository-specific factual claim MUST end with "
            "one or more valid citation tokens.\n"
            "5. Citation tokens must be copied EXACTLY from the supplied "
            "SOURCE EVIDENCE or GRAPH FACTS.\n"
            "6. Use [E#] only for source evidence.\n"
            "7. Use [G#] only for graph facts.\n"
            "8. NEVER invent another citation ID.\n"
            "9. NEVER invent URLs, links, file locations, repository URLs, "
            "or external references.\n"
            "10. Do NOT create a References section.\n"
            "11. Do NOT describe a citation as a hyperlink.\n"
            "12. Remove every claim that is not directly supported by the "
            "supplied repository context.\n"
            "13. Do not add installation instructions, roadmap, license, "
            "limitations, examples, or unrelated details unless explicitly "
            "asked by the user.\n\n"

            "A valid answer should look like:\n\n"
            "RepoBrain combines deterministic repository analysis with "
            "hybrid retrieval and grounded answer generation. [E1]\n\n"

            "NOT like:\n\n"
            "[E1] https://example.com/file\n\n"

            "If the supplied context is insufficient, return exactly:\n"
            "\"The available repository evidence is insufficient to "
            "answer this confidently.\"\n\n"

            "PREVIOUS REJECTED ANSWER\n"
            "========================\n"
            f"{original_answer}"
        )

    # =====================================================================
    # Citation validation
    # =====================================================================

    def _validate_citations(
        self,
        *,
        answer_text: str,
        citation_lookup: dict[
            str,
            object,
        ],
    ) -> tuple[
        list[AnswerCitation],
        list[str],
    ]:

        citation_ids = (
            self._citation_ids(
                answer_text
            )
        )

        valid: list[
            AnswerCitation
        ] = []

        invalid: list[str] = []

        for citation_id in (
            citation_ids
        ):

            referenced_object = (
                citation_lookup.get(
                    citation_id
                )
            )

            if (
                referenced_object
                is None
            ):

                invalid.append(
                    citation_id
                )

                continue

            # ---------------------------------------------------------
            # Source evidence
            # ---------------------------------------------------------

            if isinstance(
                referenced_object,
                EvidenceItem,
            ):

                valid.append(
                    AnswerCitation(
                        citation_id=(
                            citation_id
                        ),

                        citation_kind=(
                            AnswerCitationKind.SOURCE
                        ),

                        evidence_id=(
                            referenced_object
                            .evidence_id
                        ),

                        relative_path=(
                            referenced_object
                            .relative_path
                        ),

                        start_line=(
                            referenced_object
                            .start_line
                        ),

                        end_line=(
                            referenced_object
                            .end_line
                        ),

                        qualified_name=(
                            referenced_object
                            .qualified_name
                        ),
                    )
                )

                continue

            # ---------------------------------------------------------
            # Graph evidence
            # ---------------------------------------------------------

            if isinstance(
                referenced_object,
                AgentGraphFact,
            ):

                valid.append(
                    self._graph_citation(
                        citation_id=(
                            citation_id
                        ),

                        fact=(
                            referenced_object
                        ),
                    )
                )

                continue

            invalid.append(
                citation_id
            )

        return (
            valid,
            invalid,
        )

    # =====================================================================
    # Graph citation factory
    # =====================================================================

    @staticmethod
    def _graph_citation(
        *,
        citation_id: str,
        fact: AgentGraphFact,
    ) -> AnswerCitation:

        return AnswerCitation(
            citation_id=(
                citation_id
            ),

            citation_kind=(
                AnswerCitationKind.GRAPH
            ),

            relationship_type=(
                fact.relationship_type
            ),

            source_symbol_id=(
                fact.source_symbol_id
            ),

            source_qualified_name=(
                fact.source_qualified_name
            ),

            target_symbol_id=(
                fact.target_symbol_id
            ),

            target_qualified_name=(
                fact.target_qualified_name
            ),
        )

    # =====================================================================
    # Citation parser
    # =====================================================================

    @staticmethod
    def _citation_ids(
        answer_text: str,
    ) -> list[str]:
        """
        Extract unique E#/G# citations while preserving first-use order.
        """

        seen: set[str] = set()

        citation_ids: list[
            str
        ] = []

        for match in (
            _CITATION_PATTERN
            .finditer(
                answer_text
            )
        ):

            citation_id = (
                f"{match.group(1)}"
                f"{match.group(2)}"
            )

            if (
                citation_id
                in seen
            ):
                continue

            seen.add(
                citation_id
            )

            citation_ids.append(
                citation_id
            )

        return citation_ids