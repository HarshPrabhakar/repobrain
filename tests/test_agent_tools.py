from __future__ import annotations

from types import SimpleNamespace

from repobrain.agent import (
    AgentToolbox,
)


class FakeSymbolIndex:

    def __init__(
        self,
        symbols,
    ) -> None:
        self._symbols = symbols

    def all_symbols(
        self,
    ):
        return list(
            self._symbols
        )


class FakeHybridEngine:

    def search(
        self,
        query,
        *,
        top_k=20,
    ):
        return []


class FakeAssembler:

    def assemble(
        self,
        *,
        query,
        results,
    ):
        return SimpleNamespace(
            query=query,
            items=[],
        )


class FakeGraph:

    def get_node(
        self,
        symbol_id,
    ):
        return None

    def callers(
        self,
        symbol_id,
    ):
        return []

    def callees(
        self,
        symbol_id,
    ):
        return []


def symbol(
    symbol_id: str,
    name: str,
    qualified_name: str,
):

    return SimpleNamespace(
        symbol_id=symbol_id,
        name=name,
        qualified_name=qualified_name,
        fully_qualified_name=(
            qualified_name
        ),
    )


def toolbox(
    symbols,
):

    return AgentToolbox(
        hybrid_engine=(
            FakeHybridEngine()
        ),
        evidence_assembler=(
            FakeAssembler()
        ),
        graph=(
            FakeGraph()
        ),
        symbol_index=(
            FakeSymbolIndex(
                symbols
            )
        ),
    )


def test_resolves_exact_short_name() -> None:

    target = symbol(
        "a",
        "_hash",
        "app.Scanner._hash",
    )

    tools = toolbox(
        [
            target,
        ]
    )

    assert (
        tools.resolve_symbol(
            "_hash"
        )
        is target
    )


def test_resolves_qualified_name() -> None:

    target = symbol(
        "a",
        "scan",
        "app.Scanner.scan",
    )

    tools = toolbox(
        [
            target,
        ]
    )

    assert (
        tools.resolve_symbol(
            "app.Scanner.scan"
        )
        is target
    )


def test_ambiguous_short_name_returns_none() -> None:

    tools = toolbox(
        [
            symbol(
                "a",
                "run",
                "a.run",
            ),
            symbol(
                "b",
                "run",
                "b.run",
            ),
        ]
    )

    assert (
        tools.resolve_symbol(
            "run"
        )
        is None
    )


def test_unknown_symbol_returns_none() -> None:

    tools = toolbox(
        []
    )

    assert (
        tools.resolve_symbol(
            "missing"
        )
        is None
    )


def test_retrieve_evidence_uses_assembler() -> None:

    tools = toolbox(
        []
    )

    bundle = (
        tools.retrieve_evidence(
            "question"
        )
    )

    assert (
        bundle.query
        == "question"
    )