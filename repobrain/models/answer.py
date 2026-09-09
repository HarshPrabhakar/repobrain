from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    """
    One validated evidence citation used by an answer.
    """

    citation_id: str

    evidence_id: str

    relative_path: str | None = None

    start_line: int | None = None

    end_line: int | None = None

    qualified_name: str | None = None


class GroundedAnswer(BaseModel):
    """
    Final natural-language answer generated from RepoBrain evidence.
    """

    query: str

    answer_text: str

    model_name: str

    citations: list[
        AnswerCitation
    ] = Field(
        default_factory=list,
    )

    invalid_citation_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    evidence_items_available: int = Field(
        ge=0,
        default=0,
    )

    graph_facts_available: int = Field(
        ge=0,
        default=0,
    )

    grounded: bool = True