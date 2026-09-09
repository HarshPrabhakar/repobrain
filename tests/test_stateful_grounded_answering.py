from __future__ import annotations

from dataclasses import dataclass

import pytest

from repobrain.conversation import (
    InvestigationStateManager,
    StatefulGroundedAnsweringOrchestrator,
    StatefulRepoBrainOrchestrator,
)

from repobrain.models.agent import (
    AgentIntent,
)

from repobrain.models.conversation import (
    InvestigationState,
)


CURRENT = (
    "repobrain.ingestion.scanner."
    "RepositoryScanner._calculate_sha256"
)


# ======================================================================
# Fake Phase 7 agent result
# ======================================================================


@dataclass
class FakeAgentState:
    intent: AgentIntent


@dataclass
class FakeAgentResult:
    state: FakeAgentState

    resolved_symbol_id: str | None = None

    resolved_qualified_name: str | None = None


class RecordingAgentRunner:

    def __init__(
        self,
        *,
        result: FakeAgentResult,
    ) -> None:

        self.result = (
            result
        )

        self.queries: list[str] = []

    def __call__(
        self,
        query: str,
    ) -> FakeAgentResult:

        self.queries.append(
            query
        )

        return self.result


# ======================================================================
# Fake Phase 8 answer models
# ======================================================================


@dataclass
class FakeCitation:
    citation_id: str


@dataclass
class FakeGroundedAnswer:
    text: str

    grounded: bool

    citations: tuple[
        FakeCitation,
        ...,
    ] = ()

    invalid_citations: tuple[
        str,
        ...,
    ] = ()


class FakeAnswerGenerator:

    def __init__(
        self,
        answer: FakeGroundedAnswer,
    ) -> None:

        self.answer = (
            answer
        )

        self.results: list[
            object
        ] = []

    def generate(
        self,
        result: object,
    ) -> FakeGroundedAnswer:

        self.results.append(
            result
        )

        return self.answer


class FailingAnswerGenerator:

    def generate(
        self,
        result: object,
    ) -> object:

        raise RuntimeError(
            "answer generation failed"
        )


# ======================================================================
# Helpers
# ======================================================================


def make_stateful_agent(
    *,
    runner: RecordingAgentRunner,
    state: InvestigationState | None = None,
) -> StatefulRepoBrainOrchestrator:

    state_manager = (
        InvestigationStateManager(
            initial_state=(
                state
                if state is not None
                else InvestigationState()
            )
        )
    )

    return (
        StatefulRepoBrainOrchestrator(
            agent_runner=(
                runner
            ),
            state_manager=(
                state_manager
            ),
        )
    )


# ======================================================================
# Tests
# ======================================================================


def test_grounded_answer_metadata_is_recorded() -> None:

    agent_result = (
        FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.IMPLEMENTATION
                )
            ),
            resolved_symbol_id=(
                "symbol-sha"
            ),
            resolved_qualified_name=(
                CURRENT
            ),
        )
    )

    runner = (
        RecordingAgentRunner(
            result=(
                agent_result
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Answer [E1]",
                grounded=True,
                citations=(
                    FakeCitation(
                        citation_id="E1"
                    ),
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    turn = orchestrator.run(
        "How is the hash calculated?"
    )

    assert (
        turn.state_after
        .last_grounded
        is True
    )

    assert (
        turn.state_after
        .last_citation_ids
        == (
            "E1",
        )
    )

    assert (
        turn.state_after
        .last_invalid_citation_ids
        == ()
    )


def test_grounding_metadata_updates_history_turn() -> None:

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.GENERAL
                    )
                )
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Answer [E1]",
                grounded=True,
                citations=(
                    FakeCitation(
                        "E1"
                    ),
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    orchestrator.run(
        "What is RepoBrain?"
    )

    latest = (
        orchestrator
        .state
        .current_turn
    )

    assert latest is not None

    assert (
        latest.grounded
        is True
    )

    assert (
        latest.citation_ids
        == (
            "E1",
        )
    )


def test_invalid_citations_are_recorded() -> None:

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.GENERAL
                    )
                )
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Bad answer [E99]",
                grounded=False,
                citations=(),
                invalid_citations=(
                    "E99",
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    orchestrator.run(
        "Explain this repository."
    )

    assert (
        orchestrator
        .state
        .last_grounded
        is False
    )

    assert (
        orchestrator
        .state
        .last_citation_ids
        == ()
    )

    assert (
        orchestrator
        .state
        .last_invalid_citation_ids
        == (
            "E99",
        )
    )


def test_duplicate_citations_are_deduplicated() -> None:

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.GENERAL
                    )
                )
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Answer",
                grounded=True,
                citations=(
                    FakeCitation("E1"),
                    FakeCitation("E1"),
                    FakeCitation("G1"),
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    orchestrator.run(
        "Explain RepoBrain."
    )

    assert (
        orchestrator
        .state
        .last_citation_ids
        == (
            "E1",
            "G1",
        )
    )


def test_answer_generator_receives_agent_result() -> None:

    agent_result = (
        FakeAgentResult(
            state=FakeAgentState(
                intent=(
                    AgentIntent.GENERAL
                )
            )
        )
    )

    runner = (
        RecordingAgentRunner(
            result=(
                agent_result
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Answer",
                grounded=False,
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    orchestrator.run(
        "What is RepoBrain?"
    )

    assert len(
        answer_generator.results
    ) == 1

    assert (
        answer_generator.results[0]
        is agent_result
    )


def test_followup_is_resolved_before_grounded_answering() -> None:

    initial_state = (
        InvestigationState(
            turn_number=1,
            current_symbol_id=(
                "symbol-sha"
            ),
            current_qualified_name=(
                CURRENT
            ),
        )
    )

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.CALLEES
                    )
                )
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Graph answer [G1]",
                grounded=True,
                citations=(
                    FakeCitation(
                        "G1"
                    ),
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                    state=(
                        initial_state
                    ),
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    turn = orchestrator.run(
        "What does it call?"
    )

    assert (
        runner.queries
        == [
            f"What does {CURRENT} call?"
        ]
    )

    assert (
        turn.resolved_query
        == (
            f"What does {CURRENT} call?"
        )
    )

    assert (
        turn.state_after
        .last_citation_ids
        == (
            "G1",
        )
    )


def test_answer_failure_rolls_back_state() -> None:

    initial_state = (
        InvestigationState(
            turn_number=1,
            current_symbol_id=(
                "symbol-sha"
            ),
            current_qualified_name=(
                CURRENT
            ),
        )
    )

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.CALLEES
                    )
                )
            )
        )
    )

    stateful_agent = (
        make_stateful_agent(
            runner=runner,
            state=(
                initial_state
            ),
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                stateful_agent
            ),
            answer_generator=(
                FailingAnswerGenerator()
            ),
        )
    )

    before = (
        orchestrator.state
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "answer generation failed"
        ),
    ):

        orchestrator.run(
            "What does it call?"
        )

    assert (
        orchestrator.state
        == before
    )

    assert (
        orchestrator
        .state
        .turn_number
        == 1
    )


def test_clear_resets_grounding_history() -> None:

    runner = (
        RecordingAgentRunner(
            result=FakeAgentResult(
                state=FakeAgentState(
                    intent=(
                        AgentIntent.GENERAL
                    )
                )
            )
        )
    )

    answer_generator = (
        FakeAnswerGenerator(
            FakeGroundedAnswer(
                text="Answer [E1]",
                grounded=True,
                citations=(
                    FakeCitation(
                        "E1"
                    ),
                ),
            )
        )
    )

    orchestrator = (
        StatefulGroundedAnsweringOrchestrator(
            stateful_agent=(
                make_stateful_agent(
                    runner=runner,
                )
            ),
            answer_generator=(
                answer_generator
            ),
        )
    )

    orchestrator.run(
        "What is RepoBrain?"
    )

    state = (
        orchestrator.clear()
    )

    assert (
        state.turn_number
        == 0
    )

    assert (
        state.history
        == ()
    )

    assert (
        state.last_grounded
        is None
    )

    assert (
        state.last_citation_ids
        == ()
    )


def test_update_grounding_without_turn_is_rejected() -> None:

    manager = (
        InvestigationStateManager()
    )

    with pytest.raises(
        ValueError,
        match=(
            "cannot update grounding without a recorded turn"
        ),
    ):

        manager.update_last_turn_grounding(
            grounded=True,
            citation_ids=(
                "E1",
            ),
        )