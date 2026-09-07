from __future__ import annotations

from repobrain.indexing.symbol_index import (
    SymbolIndex,
)

from repobrain.models.symbols import (
    CodeSymbol,
    SymbolType,
)


def make_symbol(
    symbol_id: str,
    name: str,
    qualified_name: str,
    symbol_type: SymbolType,
    module: str,
    parent_symbol_id: str | None = None,
    start_line: int = 1,
) -> CodeSymbol:

    return CodeSymbol(
        symbol_id=symbol_id,

        repository_id="repo_test",
        file_id="file_test",

        name=name,

        qualified_name=qualified_name,

        fully_qualified_name=(
            qualified_name
        ),

        symbol_type=symbol_type,

        module=module,

        start_line=start_line,
        end_line=start_line,

        parent_symbol_id=(
            parent_symbol_id
        ),
    )


def test_index_lookup_by_id() -> None:

    symbol = make_symbol(
        symbol_id="sym_a",

        name="login",

        qualified_name=(
            "app.auth.login"
        ),

        symbol_type=SymbolType.FUNCTION,

        module="app.auth",
    )

    index = SymbolIndex(
        [symbol]
    )

    assert (
        index.get_by_id("sym_a")
        == symbol
    )


def test_find_by_qualified_name() -> None:

    symbol = make_symbol(
        symbol_id="sym_a",

        name="login",

        qualified_name=(
            "app.auth.login"
        ),

        symbol_type=SymbolType.FUNCTION,

        module="app.auth",
    )

    index = SymbolIndex(
        [symbol]
    )

    matches = (
        index.find_by_qualified_name(
            "app.auth.login"
        )
    )

    assert matches == [symbol]


def test_find_unique_symbol() -> None:

    symbol = make_symbol(
        symbol_id="sym_a",

        name="login",

        qualified_name=(
            "app.auth.login"
        ),

        symbol_type=SymbolType.FUNCTION,

        module="app.auth",
    )

    index = SymbolIndex(
        [symbol]
    )

    assert (
        index.find_unique_by_qualified_name(
            "app.auth.login"
        )
        == symbol
    )


def test_duplicate_qualified_name_is_not_unique() -> None:

    first = make_symbol(
        symbol_id="sym_a",

        name="login",

        qualified_name=(
            "app.auth.login"
        ),

        symbol_type=SymbolType.FUNCTION,

        module="app.auth",

        start_line=1,
    )

    second = make_symbol(
        symbol_id="sym_b",

        name="login",

        qualified_name=(
            "app.auth.login"
        ),

        symbol_type=SymbolType.FUNCTION,

        module="app.auth",

        start_line=10,
    )

    index = SymbolIndex(
        [
            first,
            second,
        ]
    )

    assert (
        index.find_unique_by_qualified_name(
            "app.auth.login"
        )
        is None
    )


def test_find_by_short_name() -> None:

    one = make_symbol(
        symbol_id="sym_a",
        name="save",
        qualified_name="a.save",
        symbol_type=SymbolType.FUNCTION,
        module="a",
    )

    two = make_symbol(
        symbol_id="sym_b",
        name="save",
        qualified_name="b.save",
        symbol_type=SymbolType.FUNCTION,
        module="b",
    )

    index = SymbolIndex(
        [
            one,
            two,
        ]
    )

    matches = (
        index.find_by_name(
            "save"
        )
    )

    assert len(matches) == 2


def test_find_symbols_in_module() -> None:

    one = make_symbol(
        symbol_id="sym_a",
        name="one",
        qualified_name="app.one",
        symbol_type=SymbolType.FUNCTION,
        module="app",
    )

    two = make_symbol(
        symbol_id="sym_b",
        name="two",
        qualified_name="app.two",
        symbol_type=SymbolType.FUNCTION,
        module="app",
    )

    index = SymbolIndex(
        [
            one,
            two,
        ]
    )

    assert len(
        index.find_in_module(
            "app"
        )
    ) == 2


def test_find_containing_class() -> None:

    class_symbol = make_symbol(
        symbol_id="class_1",

        name="Service",

        qualified_name=(
            "app.Service"
        ),

        symbol_type=SymbolType.CLASS,

        module="app",
    )

    method = make_symbol(
        symbol_id="method_1",

        name="execute",

        qualified_name=(
            "app.Service.execute"
        ),

        symbol_type=SymbolType.METHOD,

        module="app",

        parent_symbol_id="class_1",
    )

    index = SymbolIndex(
        [
            class_symbol,
            method,
        ]
    )

    assert (
        index.find_containing_class(
            method
        )
        == class_symbol
    )