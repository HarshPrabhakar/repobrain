from __future__ import annotations

from difflib import SequenceMatcher

from repobrain.indexing.symbol_index import SymbolIndex
from repobrain.models.symbols import (
    CodeSymbol,
    SymbolSearchResult,
    SymbolType,
)


class SymbolSearchEngine:
    """
    Deterministic Phase 3A symbol search engine.

    Search priority:

    1. Exact qualified-name match
    2. Exact fully-qualified-name match
    3. Exact short-name match
    4. Qualified-name suffix match
    5. Short-name prefix match
    6. Short-name substring match
    7. Qualified-name substring match
    8. Fuzzy short-name match

    This engine performs lexical symbol search only.
    It does not perform semantic search or LLM reasoning.
    """

    def __init__(
        self,
        symbol_index: SymbolIndex,
    ) -> None:
        self.index = symbol_index

    # =========================================================
    # Public API
    # =========================================================

    def search(
        self,
        query: str,
        *,
        symbol_type: SymbolType | None = None,
        module: str | None = None,
        top_k: int = 10,
    ) -> list[SymbolSearchResult]:
        """
        Search repository symbols.

        Results are ranked deterministically.

        Parameters
        ----------
        query:
            Symbol name or qualified-name fragment.

        symbol_type:
            Optional symbol type filter.

        module:
            Optional exact module filter.

        top_k:
            Maximum number of results returned.
        """

        normalized_query = self._normalize(query)

        if not normalized_query:
            return []

        if top_k <= 0:
            return []

        candidates = self._all_symbols()

        results: list[SymbolSearchResult] = []

        for symbol in candidates:

            if (
                symbol_type is not None
                and symbol.symbol_type != symbol_type
            ):
                continue

            if (
                module is not None
                and symbol.module != module
            ):
                continue

            score_data = self._score_symbol(
                query=normalized_query,
                symbol=symbol,
            )

            if score_data is None:
                continue

            score, match_type, reason = score_data

            results.append(
                SymbolSearchResult(
                    symbol_id=symbol.symbol_id,
                    name=symbol.name,
                    qualified_name=symbol.qualified_name,
                    fully_qualified_name=(
                        symbol.fully_qualified_name
                    ),
                    symbol_type=symbol.symbol_type,
                    module=symbol.module,
                    file_id=symbol.file_id,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    score=score,
                    match_type=match_type,
                    reason=reason,
                )
            )

        results.sort(
            key=self._sort_key
        )

        return results[:top_k]

    def find_definition(
        self,
        query: str,
        *,
        symbol_type: SymbolType | None = None,
    ) -> list[SymbolSearchResult]:
        """
        Return the strongest matching repository definitions.

        If several symbols have the same best score, all of them
        are preserved so ambiguous short names are not arbitrarily
        resolved.
        """

        results = self.search(
            query=query,
            symbol_type=symbol_type,
            top_k=50,
        )

        if not results:
            return []

        best_score = results[0].score

        return [
            result
            for result in results
            if result.score == best_score
        ]

    def find_exact(
        self,
        query: str,
    ) -> list[CodeSymbol]:
        """
        Perform exact deterministic symbol lookup.

        Matches against:

        - symbol.name
        - symbol.qualified_name
        - symbol.fully_qualified_name

        Unlike search(), this method performs no fuzzy or partial
        matching.
        """

        normalized_query = query.strip()

        if not normalized_query:
            return []

        results: list[CodeSymbol] = []

        seen: set[str] = set()

        for symbol in self._all_symbols():

            if (
                symbol.name == normalized_query
                or symbol.qualified_name == normalized_query
                or symbol.fully_qualified_name
                == normalized_query
            ):

                if symbol.symbol_id in seen:
                    continue

                seen.add(
                    symbol.symbol_id
                )

                results.append(symbol)

        return sorted(
            results,
            key=lambda item: (
                item.qualified_name.casefold(),
                item.start_line,
                item.symbol_id,
            ),
        )

    # =========================================================
    # Symbol scoring
    # =========================================================

    def _score_symbol(
        self,
        query: str,
        symbol: CodeSymbol,
    ) -> tuple[
        float,
        str,
        str,
    ] | None:
        """
        Calculate deterministic lexical relevance for one symbol.
        """

        short_name = self._normalize(
            symbol.name
        )

        qualified = self._normalize(
            symbol.qualified_name
        )

        fully_qualified = self._normalize(
            symbol.fully_qualified_name
        )

        # -----------------------------------------------------
        # 1. Exact qualified-name match
        # -----------------------------------------------------

        if query == qualified:
            return (
                1.0,
                "EXACT_QUALIFIED",
                "Exact qualified-name match",
            )

        # -----------------------------------------------------
        # 2. Exact fully-qualified-name match
        # -----------------------------------------------------

        if query == fully_qualified:
            return (
                1.0,
                "EXACT_FULLY_QUALIFIED",
                "Exact fully-qualified name match",
            )

        # -----------------------------------------------------
        # 3. Exact short-name match
        # -----------------------------------------------------

        if query == short_name:
            return (
                0.95,
                "EXACT_NAME",
                "Exact short-name match",
            )

        # -----------------------------------------------------
        # 4. Qualified suffix
        #
        # Example:
        #
        # RepositoryScanner.scan
        #
        # matches:
        #
        # repobrain.ingestion.scanner.RepositoryScanner.scan
        # -----------------------------------------------------

        qualified_suffix = f".{query}"

        if qualified.endswith(
            qualified_suffix
        ):
            return (
                0.90,
                "QUALIFIED_SUFFIX",
                (
                    "Query matches the end "
                    "of the qualified name"
                ),
            )

        # -----------------------------------------------------
        # 5. Short-name prefix
        #
        # parse
        # ->
        # parse_source
        # -----------------------------------------------------

        if short_name.startswith(query):

            ratio = (
                len(query)
                / max(
                    len(short_name),
                    1,
                )
            )

            score = (
                0.78
                + (
                    0.10
                    * ratio
                )
            )

            return (
                min(
                    score,
                    0.89,
                ),
                "NAME_PREFIX",
                "Short name starts with query",
            )

        # -----------------------------------------------------
        # 6. Short-name substring
        # -----------------------------------------------------

        if query in short_name:

            ratio = (
                len(query)
                / max(
                    len(short_name),
                    1,
                )
            )

            score = (
                0.65
                + (
                    0.12
                    * ratio
                )
            )

            return (
                min(
                    score,
                    0.79,
                ),
                "NAME_SUBSTRING",
                "Query appears in short name",
            )

        # -----------------------------------------------------
        # 7. Qualified-name substring
        # -----------------------------------------------------

        if query in qualified:
            return (
                0.62,
                "QUALIFIED_SUBSTRING",
                (
                    "Query appears in "
                    "qualified name"
                ),
            )

        # -----------------------------------------------------
        # 8. Fuzzy short-name similarity
        # -----------------------------------------------------

        fuzzy_similarity = SequenceMatcher(
            None,
            query,
            short_name,
        ).ratio()

        if fuzzy_similarity >= 0.65:

            score = (
                0.40
                + (
                    fuzzy_similarity
                    * 0.20
                )
            )

            return (
                min(
                    score,
                    0.59,
                ),
                "FUZZY_NAME",
                (
                    "Fuzzy textual similarity "
                    "with short name"
                ),
            )

        return None

    # =========================================================
    # Internal helpers
    # =========================================================

    def _all_symbols(
        self,
    ) -> list[CodeSymbol]:
        """
        Retrieve all repository symbols through the public
        SymbolIndex interface.
        """

        return self.index.all_symbols()

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:
        """
        Normalize search text for case-insensitive matching.
        """

        return value.strip().casefold()

    @staticmethod
    def _sort_key(
        result: SymbolSearchResult,
    ) -> tuple:
        """
        Deterministic result ordering.

        Higher score comes first.

        When scores are equal:

        CLASS
        FUNCTION
        METHOD
        MODULE

        are preferred in that order.
        """

        type_priority = {
            SymbolType.CLASS: 0,
            SymbolType.FUNCTION: 1,
            SymbolType.METHOD: 2,
            SymbolType.MODULE: 3,
        }

        return (
            -result.score,
            type_priority.get(
                result.symbol_type,
                99,
            ),
            result.qualified_name.casefold(),
            result.start_line,
            result.symbol_id,
        )