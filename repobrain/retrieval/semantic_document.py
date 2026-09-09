from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from repobrain.models.retrieval import (
    CodeChunk,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    RelationshipType,
)


_IDENTIFIER_PATTERN = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z])
    |
    [A-Z]?[a-z]+[0-9]*
    |
    [A-Z]+[0-9]*
    |
    [0-9]+
    """,
    re.VERBOSE,
)


class SemanticCodeDocumentBuilder:
    """
    Build deterministic semantic documents for code retrieval.

    Phase 3C.2 enriches raw source with repository intelligence
    that RepoBrain has already extracted deterministically.

    No LLM is involved.

    Inputs:
        CodeChunk
        CodeSymbol
        CodeRelationship

    Output:
        text suitable for semantic embedding
    """

    def __init__(
        self,
        symbols: Iterable[CodeSymbol],
        relationships: Iterable[CodeRelationship],
    ) -> None:

        self._symbols_by_id: dict[
            str,
            CodeSymbol,
        ] = {}

        self._relationships_by_source: dict[
            str,
            list[CodeRelationship],
        ] = defaultdict(list)

        for symbol in symbols:
            self._symbols_by_id[
                symbol.symbol_id
            ] = symbol

        for relationship in relationships:

            source_symbol_id = (
                relationship.source_symbol_id
            )

            if source_symbol_id is None:
                continue

            self._relationships_by_source[
                source_symbol_id
            ].append(
                relationship
            )

    # =========================================================
    # Public API
    # =========================================================

    def build(
        self,
        chunk: CodeChunk,
    ) -> str:
        """
        Build semantic embedding text for one chunk.

        Chunks without a known symbol remain raw text.

        This is important for:
            README
            TOML
            config files
            generic non-symbol files
        """

        if chunk.symbol_id is None:
            return chunk.text

        symbol = self._symbols_by_id.get(
            chunk.symbol_id
        )

        if symbol is None:
            return chunk.text

        return self._build_symbol_document(
            chunk=chunk,
            symbol=symbol,
        )

    # =========================================================
    # Symbol document
    # =========================================================

    def _build_symbol_document(
        self,
        *,
        chunk: CodeChunk,
        symbol: CodeSymbol,
    ) -> str:

        sections: list[str] = []

        # -----------------------------------------------------
        # Symbol identity
        # -----------------------------------------------------

        sections.append(
            f"Symbol: {symbol.name}"
        )

        sections.append(
            (
                "Qualified name: "
                f"{symbol.qualified_name}"
            )
        )

        sections.append(
            (
                "Type: "
                f"{symbol.symbol_type.value.lower()}"
            )
        )

        # -----------------------------------------------------
        # Identifier components
        # -----------------------------------------------------

        identifier_terms = (
            self._identifier_terms(
                symbol.name
            )
        )

        if identifier_terms:

            sections.append(
                (
                    "Identifier terms: "
                    + " ".join(
                        identifier_terms
                    )
                )
            )

        # -----------------------------------------------------
        # Signature
        # -----------------------------------------------------

        if symbol.signature:

            sections.append(
                (
                    "Signature: "
                    f"{symbol.signature}"
                )
            )

        # -----------------------------------------------------
        # Documentation
        # -----------------------------------------------------

        if symbol.docstring:

            normalized_docstring = (
                self._normalize_text(
                    symbol.docstring
                )
            )

            if normalized_docstring:

                sections.append(
                    (
                        "Documentation: "
                        f"{normalized_docstring}"
                    )
                )

        # -----------------------------------------------------
        # Async information
        # -----------------------------------------------------

        if symbol.is_async:

            sections.append(
                "Async: yes"
            )

        # -----------------------------------------------------
        # Decorators
        # -----------------------------------------------------

        decorators = self._decorators(
            symbol
        )

        if decorators:

            sections.append(
                "Decorators:"
            )

            sections.extend(
                f"- {decorator}"
                for decorator
                in decorators
            )

        # -----------------------------------------------------
        # Calls
        # -----------------------------------------------------

        calls = self._calls(
            symbol.symbol_id
        )

        if calls:

            sections.append(
                "Calls:"
            )

            sections.extend(
                f"- {call}"
                for call
                in calls
            )

        # -----------------------------------------------------
        # Source
        # -----------------------------------------------------

        sections.append(
            "Source:"
        )

        sections.append(
            chunk.text
        )

        return "\n".join(
            sections
        )

    # =========================================================
    # Relationship extraction
    # =========================================================

    def _calls(
        self,
        symbol_id: str,
    ) -> list[str]:

        calls: set[str] = set()

        relationships = (
            self._relationships_by_source.get(
                symbol_id,
                [],
            )
        )

        for relationship in relationships:

            if (
                relationship.relationship_type
                != RelationshipType.CALLS
            ):
                continue

            target = (
                relationship.target_qualified_name
            )

            if not target:
                continue

            target = target.strip()

            if target:
                calls.add(
                    target
                )

        return sorted(
            calls,
            key=str.casefold,
        )

    # =========================================================
    # Decorators
    # =========================================================

    @staticmethod
    def _decorators(
        symbol: CodeSymbol,
    ) -> list[str]:

        decorators = {
            decorator.strip()
            for decorator
            in symbol.decorators
            if decorator.strip()
        }

        return sorted(
            decorators,
            key=str.casefold,
        )

    # =========================================================
    # Identifier decomposition
    # =========================================================

    @staticmethod
    def _identifier_terms(
        identifier: str,
    ) -> list[str]:
        """
        Deterministically expose useful identifier components.

        Example:

            _calculate_sha256

        becomes:

            calculate
            sha256

        Example:

            buildFileMetadata

        becomes:

            build
            file
            metadata
        """

        cleaned = (
            identifier
            .strip()
            .strip("_")
        )

        if not cleaned:
            return []

        terms: list[str] = []

        for underscore_part in (
            cleaned.split("_")
        ):

            if not underscore_part:
                continue

            camel_parts = (
                _IDENTIFIER_PATTERN.findall(
                    underscore_part
                )
            )

            if camel_parts:

                terms.extend(
                    part.casefold()
                    for part
                    in camel_parts
                )

            else:

                terms.append(
                    underscore_part.casefold()
                )

        return list(
            dict.fromkeys(
                terms
            )
        )

    # =========================================================
    # Text normalization
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        return " ".join(
            text.split()
        )