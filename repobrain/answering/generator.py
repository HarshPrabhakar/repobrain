from __future__ import annotations

import re

from repobrain.answering.prompt_builder import (
    GroundedPromptBuilder,
)

from repobrain.llm.base import (
    LLMProvider,
)

from repobrain.models.agent import (
    AgentRunResult,
)

from repobrain.models.answer import (
    AnswerCitation,
    GroundedAnswer,
)


_CITATION_PATTERN = re.compile(
    r"\[E(\d+)\]"
)


class GroundedAnswerGenerator:
    """
    Generate natural-language answers using only assembled
    repository evidence.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        prompt_builder: (
            GroundedPromptBuilder
            | None
        ) = None,
    ) -> None:

        self.provider = (
            provider
        )

        self.prompt_builder = (
            prompt_builder
            or GroundedPromptBuilder()
        )

    def generate(
        self,
        result: AgentRunResult,
    ) -> GroundedAnswer:

        prompt, citation_lookup = (
            self.prompt_builder.build(
                result
            )
        )

        answer_text = (
            self.provider.generate(
                instructions=(
                    self.prompt_builder
                    .SYSTEM_INSTRUCTIONS
                ),
                input_text=prompt,
            )
        )

        citation_ids = (
            self._citation_ids(
                answer_text
            )
        )

        valid_citations: list[
            AnswerCitation
        ] = []

        invalid_citations: list[
            str
        ] = []

        for citation_id in citation_ids:

            item = (
                citation_lookup.get(
                    citation_id
                )
            )

            if item is None:

                invalid_citations.append(
                    citation_id
                )

                continue

            valid_citations.append(
                AnswerCitation(
                    citation_id=(
                        citation_id
                    ),
                    evidence_id=(
                        item.evidence_id
                    ),
                    relative_path=(
                        item.relative_path
                    ),
                    start_line=(
                        item.start_line
                    ),
                    end_line=(
                        item.end_line
                    ),
                    qualified_name=(
                        item.qualified_name
                    ),
                )
            )

        bundle = (
            result.evidence
        )

        has_repository_context = bool(
            (
                bundle
                and bundle.items
            )
            or result.graph_facts
        )

        grounded = (
            has_repository_context
            and bool(valid_citations)
            and not invalid_citations   
        )

        return GroundedAnswer(
            query=(
                result.state.query
            ),

            answer_text=(
                answer_text
            ),

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
                len(
                    bundle.items
                )
                if bundle
                is not None
                else 0
            ),

            graph_facts_available=(
                len(
                    result.graph_facts
                )
            ),

            grounded=(
                grounded
            ),
        )

    @staticmethod
    def _citation_ids(
        answer_text: str,
    ) -> list[str]:

        seen: set[str] = set()

        result: list[str] = []

        for match in (
            _CITATION_PATTERN
            .finditer(
                answer_text
            )
        ):

            citation_id = (
                f"E{match.group(1)}"
            )

            if citation_id in seen:
                continue

            seen.add(
                citation_id
            )

            result.append(
                citation_id
            )

        return result