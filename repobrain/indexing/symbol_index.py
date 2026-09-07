from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from repobrain.models.symbols import (
    CodeSymbol,
    SymbolType,
)


class SymbolIndex:
    """
    Deterministic repository-wide symbol lookup index.

    The SymbolIndex does not perform fuzzy matching,
    semantic search, embeddings, or LLM reasoning.

    It provides exact structural lookup over symbols
    discovered during Phase 2 AST extraction.
    """

    def __init__(
        self,
        symbols: Iterable[CodeSymbol],
    ) -> None:
        """
        Build a symbol index from repository symbols.
        """

        self._by_id: dict[
            str,
            CodeSymbol,
        ] = {}

        self._by_qualified_name: dict[
            str,
            list[CodeSymbol],
        ] = defaultdict(list)

        self._by_name: dict[
            str,
            list[CodeSymbol],
        ] = defaultdict(list)

        self._by_module: dict[
            str,
            list[CodeSymbol],
        ] = defaultdict(list)

        for symbol in symbols:
            self.add(symbol)

    # =========================================================
    # Construction
    # =========================================================

    def add(
        self,
        symbol: CodeSymbol,
    ) -> None:
        """
        Add one symbol to the index.

        Symbols are indexed by:

        - stable symbol ID
        - exact qualified name
        - short name
        - module
        """

        existing = self._by_id.get(
            symbol.symbol_id
        )

        if existing is not None:
            # If the same exact symbol is added again,
            # avoid duplicating secondary indexes.
            if existing == symbol:
                return

            raise ValueError(
                "Duplicate symbol_id detected with "
                "different symbol data: "
                f"{symbol.symbol_id}"
            )

        self._by_id[
            symbol.symbol_id
        ] = symbol

        self._by_qualified_name[
            symbol.qualified_name
        ].append(symbol)

        self._by_name[
            symbol.name
        ].append(symbol)

        self._by_module[
            symbol.module
        ].append(symbol)

    # =========================================================
    # ID lookup
    # =========================================================

    def get_by_id(
        self,
        symbol_id: str,
    ) -> CodeSymbol | None:
        """
        Return one symbol by stable symbol ID.
        """

        return self._by_id.get(
            symbol_id
        )

    # =========================================================
    # Qualified-name lookup
    # =========================================================

    def find_by_qualified_name(
        self,
        qualified_name: str,
    ) -> list[CodeSymbol]:
        """
        Find all symbols matching an exact qualified name.

        A list is returned because Python permits definitions
        to be redefined in the same scope.

        Example:

        def process():
            pass

        def process():
            pass

        Both may have the same qualified name but different
        symbol IDs and source locations.
        """

        return list(
            self._by_qualified_name.get(
                qualified_name,
                [],
            )
        )

    def find_unique_by_qualified_name(
        self,
        qualified_name: str,
    ) -> CodeSymbol | None:
        """
        Return a symbol only when the exact qualified name
        identifies exactly one repository symbol.

        Ambiguous matches intentionally return None.
        """

        matches = (
            self.find_by_qualified_name(
                qualified_name
            )
        )

        if len(matches) != 1:
            return None

        return matches[0]

    # =========================================================
    # Short-name lookup
    # =========================================================

    def find_by_name(
        self,
        name: str,
    ) -> list[CodeSymbol]:
        """
        Find all repository symbols sharing an exact short name.

        Example:

        RepositoryScanner.scan
        SecurityScanner.scan

        find_by_name("scan")

        returns both symbols.
        """

        return list(
            self._by_name.get(
                name,
                [],
            )
        )

    # =========================================================
    # Module lookup
    # =========================================================

    def find_in_module(
        self,
        module: str,
    ) -> list[CodeSymbol]:
        """
        Return all symbols belonging to an exact module.
        """

        return list(
            self._by_module.get(
                module,
                [],
            )
        )

    # =========================================================
    # Structural context
    # =========================================================

    def get_parent(
        self,
        symbol: CodeSymbol,
    ) -> CodeSymbol | None:
        """
        Return the symbol's direct structural parent.

        Examples:

        METHOD
        -> CLASS

        CLASS
        -> MODULE

        nested FUNCTION
        -> FUNCTION or METHOD
        """

        if symbol.parent_symbol_id is None:
            return None

        return self.get_by_id(
            symbol.parent_symbol_id
        )

    def find_containing_class(
        self,
        symbol: CodeSymbol,
    ) -> CodeSymbol | None:
        """
        Walk parent relationships until the nearest containing
        class is found.

        This also supports nested functions inside methods.

        Example:

        class Service:

            def outer(self):

                def inner():
                    self.execute()

        Structural chain:

        inner
          ->
        outer
          ->
        Service

        The returned containing class is Service.
        """

        current: CodeSymbol | None = (
            symbol
        )

        visited: set[str] = set()

        while current is not None:

            if (
                current.symbol_id
                in visited
            ):
                # Defensive protection against
                # malformed cyclic parent links.
                return None

            visited.add(
                current.symbol_id
            )

            if (
                current.symbol_type
                == SymbolType.CLASS
            ):
                return current

            current = self.get_parent(
                current
            )

        return None

    # =========================================================
    # Repository-wide access
    # =========================================================

    def all_symbols(
        self,
    ) -> list[CodeSymbol]:
        """
        Return all indexed symbols.

        Python dictionaries preserve insertion order, so this
        keeps the deterministic order in which symbols were
        originally added to the index.

        Phase 3A SymbolSearchEngine uses this public method
        instead of accessing the private _by_id dictionary.
        """

        return list(
            self._by_id.values()
        )

    # =========================================================
    # Statistics
    # =========================================================

    def __len__(
        self,
    ) -> int:
        """
        Return the number of indexed symbols.
        """

        return len(
            self._by_id
        )