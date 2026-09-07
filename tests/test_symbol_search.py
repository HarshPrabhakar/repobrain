from __future__ import annotations

from repobrain.indexing import (
    SymbolIndex,
)

from repobrain.models.symbols import (
    CodeSymbol,
    SymbolType,
)

from repobrain.retrieval import (
    SymbolSearchEngine,
)


def make_symbol(
    symbol_id: str,
    name: str,
    qualified_name: str,
    symbol_type: SymbolType,
    module: str,
    start_line: int = 1,
) -> CodeSymbol:

    return CodeSymbol(
        symbol_id=symbol_id,

        repository_id="repo_test",

        file_id=f"file_{symbol_id}",

        name=name,

        qualified_name=qualified_name,

        fully_qualified_name=(
            qualified_name
        ),

        symbol_type=symbol_type,

        module=module,

        start_line=start_line,
        end_line=start_line + 5,
    )


def build_engine() -> SymbolSearchEngine:

    symbols = [
        make_symbol(
            symbol_id="sym_repo_scanner",

            name="RepositoryScanner",

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner"
            ),

            symbol_type=SymbolType.CLASS,

            module=(
                "repobrain.ingestion.scanner"
            ),
        ),

        make_symbol(
            symbol_id="sym_scan",

            name="scan",

            qualified_name=(
                "repobrain.ingestion.scanner."
                "RepositoryScanner.scan"
            ),

            symbol_type=SymbolType.METHOD,

            module=(
                "repobrain.ingestion.scanner"
            ),

            start_line=50,
        ),

        make_symbol(
            symbol_id="sym_parse_source",

            name="parse_source",

            qualified_name=(
                "repobrain.parsing.python_ast."
                "PythonASTParser.parse_source"
            ),

            symbol_type=SymbolType.METHOD,

            module=(
                "repobrain.parsing.python_ast"
            ),

            start_line=100,
        ),

        make_symbol(
            symbol_id="sym_run",

            name="run",

            qualified_name=(
                "app.worker.run"
            ),

            symbol_type=SymbolType.FUNCTION,

            module="app.worker",
        ),

        make_symbol(
            symbol_id="sym_run_two",

            name="run",

            qualified_name=(
                "app.tasks.run"
            ),

            symbol_type=SymbolType.FUNCTION,

            module="app.tasks",
        ),
    ]

    index = SymbolIndex(
        symbols
    )

    return SymbolSearchEngine(
        index
    )


def test_exact_qualified_name_ranked_first() -> None:

    engine = build_engine()

    results = engine.search(
        (
            "repobrain.ingestion.scanner."
            "RepositoryScanner.scan"
        )
    )

    assert results

    assert (
        results[0].symbol_id
        == "sym_scan"
    )

    assert (
        results[0].score
        == 1.0
    )


def test_exact_short_name() -> None:

    engine = build_engine()

    results = engine.search(
        "RepositoryScanner"
    )

    assert results

    assert (
        results[0].symbol_id
        == "sym_repo_scanner"
    )

    assert (
        results[0].match_type
        == "EXACT_NAME"
    )


def test_case_insensitive_search() -> None:

    engine = build_engine()

    results = engine.search(
        "repositoryscanner"
    )

    assert results

    assert (
        results[0].symbol_id
        == "sym_repo_scanner"
    )


def test_qualified_suffix_search() -> None:

    engine = build_engine()

    results = engine.search(
        "RepositoryScanner.scan"
    )

    assert results

    assert (
        results[0].symbol_id
        == "sym_scan"
    )


def test_partial_symbol_name() -> None:

    engine = build_engine()

    results = engine.search(
        "parse"
    )

    assert results

    assert (
        results[0].symbol_id
        == "sym_parse_source"
    )


def test_symbol_type_filter() -> None:

    engine = build_engine()

    results = engine.search(
        "run",

        symbol_type=(
            SymbolType.FUNCTION
        ),
    )

    assert len(results) == 2

    assert all(
        result.symbol_type
        == SymbolType.FUNCTION
        for result in results
    )


def test_module_filter() -> None:

    engine = build_engine()

    results = engine.search(
        "run",

        module="app.worker",
    )

    assert len(results) == 1

    assert (
        results[0].qualified_name
        == "app.worker.run"
    )


def test_ambiguous_short_names_preserved() -> None:

    engine = build_engine()

    results = engine.search(
        "run"
    )

    exact_name_results = [
        result
        for result in results
        if result.match_type
        == "EXACT_NAME"
    ]

    assert (
        len(
            exact_name_results
        )
        == 2
    )


def test_find_definition_preserves_ambiguity() -> None:

    engine = build_engine()

    results = (
        engine.find_definition(
            "run"
        )
    )

    assert len(results) == 2


def test_find_definition_exact_qualified() -> None:

    engine = build_engine()

    results = (
        engine.find_definition(
            (
                "repobrain.parsing.python_ast."
                "PythonASTParser.parse_source"
            )
        )
    )

    assert len(results) == 1

    assert (
        results[0].symbol_id
        == "sym_parse_source"
    )


def test_find_exact_short_name() -> None:

    engine = build_engine()

    results = engine.find_exact(
        "run"
    )

    assert len(results) == 2


def test_find_exact_qualified_name() -> None:

    engine = build_engine()

    results = engine.find_exact(
        (
            "repobrain.ingestion.scanner."
            "RepositoryScanner.scan"
        )
    )

    assert len(results) == 1

    assert (
        results[0].symbol_id
        == "sym_scan"
    )


def test_empty_query_returns_no_results() -> None:

    engine = build_engine()

    assert (
        engine.search("   ")
        == []
    )


def test_top_k_limit() -> None:

    engine = build_engine()

    results = engine.search(
        "run",
        top_k=1,
    )

    assert len(results) == 1


def test_unknown_symbol_returns_empty() -> None:

    engine = build_engine()

    results = engine.search(
        "totally_nonexistent_symbol_xyz"
    )

    assert results == []