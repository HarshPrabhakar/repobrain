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

    The SymbolIndex does not perform fuzzy matching or semantic reasoning.

    It provides exact structural lookup over symbols discovered during
    Phase 2 AST extraction.
    """

    def __init__(
        self,
        symbols: Iterable[CodeSymbol],
    ) -> None:

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

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def add(
        self,
        symbol: CodeSymbol,
    ) -> None:
        """
        Add one symbol to the index.
        """

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

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

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

    def find_by_qualified_name(
        self,
        qualified_name: str,
    ) -> list[CodeSymbol]:
        """
        Find all symbols matching an exact qualified name.

        A list is returned because Python technically permits symbols to
        be redefined in the same scope.
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
        Return a symbol only when the qualified name uniquely identifies
        exactly one repository symbol.
        """

        matches = self.find_by_qualified_name(
            qualified_name
        )

        if len(matches) != 1:
            return None

        return matches[0]

    def find_by_name(
        self,
        name: str,
    ) -> list[CodeSymbol]:
        """
        Find repository symbols sharing the same short name.
        """

        return list(
            self._by_name.get(
                name,
                [],
            )
        )

    def find_in_module(
        self,
        module: str,
    ) -> list[CodeSymbol]:
        """
        Return all symbols belonging to a module.
        """

        return list(
            self._by_module.get(
                module,
                [],
            )
        )

    # ---------------------------------------------------------
    # Structural context
    # ---------------------------------------------------------

    def get_parent(
        self,
        symbol: CodeSymbol,
    ) -> CodeSymbol | None:
        """
        Return the symbol's direct structural parent.
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
        Walk parent relationships until the nearest containing class
        is found.

        This also works for nested functions inside methods.

        Example:

        class Service:
            def outer(self):
                def inner():
                    self.execute()

        inner -> outer -> Service
        """

        current: CodeSymbol | None = symbol

        visited: set[str] = set()

        while current is not None:

            if current.symbol_id in visited:
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

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def __len__(
        self,
    ) -> int:
        return len(
            self._by_id
        )