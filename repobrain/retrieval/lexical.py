from __future__ import annotations

import math
import re

from collections import Counter, defaultdict

from repobrain.models.retrieval import (
    CodeChunk,
    LexicalSearchResult,
)


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+"
)

_CAMEL_PATTERN = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z]|\b)
    |
    [A-Z]?[a-z]+
    |
    [A-Z]+
    |
    \d+
    """,
    re.VERBOSE,
)


class BM25Tokenizer:
    """
    Deterministic tokenizer optimized for source code.

    It keeps complete identifiers while also generating
    useful identifier sub-parts.

    Example:

        RepositoryScanner

    becomes roughly:

        repositoryscanner
        repository
        scanner

    And:

        calculate_sha256

    becomes:

        calculate_sha256
        calculate
        sha256
    """

    def tokenize(
        self,
        text: str,
    ) -> list[str]:

        tokens: list[str] = []

        for match in _TOKEN_PATTERN.finditer(
            text
        ):

            raw = match.group(0)

            normalized = (
                raw.casefold()
            )

            tokens.append(
                normalized
            )

            parts = self._identifier_parts(
                raw
            )

            for part in parts:

                normalized_part = (
                    part.casefold()
                )

                if (
                    normalized_part
                    != normalized
                ):
                    tokens.append(
                        normalized_part
                    )

        return tokens

    @staticmethod
    def _identifier_parts(
        token: str,
    ) -> list[str]:

        results: list[str] = []

        underscore_parts = [
            part
            for part in token.split("_")
            if part
        ]

        for part in underscore_parts:

            # Preserve the whole underscore-separated component.
            #
            # Example:
            # calculate_sha256
            #
            # becomes:
            # calculate
            # sha256
            results.append(part)

            camel_parts = (
                _CAMEL_PATTERN.findall(
                    part
                )
            )

            for camel_part in camel_parts:

                if camel_part == part:
                    continue

                results.append(
                    camel_part
                )

        return list(
            dict.fromkeys(
                results
            )
        )


class BM25Index:
    """
    In-memory BM25 index over repository CodeChunk objects.

    Formula follows the standard Okapi BM25 structure.

    This implementation is deliberately small and transparent.
    """

    def __init__(
        self,
        chunks: list[CodeChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: BM25Tokenizer | None = None,
    ) -> None:

        self.chunks = list(
            chunks
        )

        self.k1 = k1
        self.b = b

        self.tokenizer = (
            tokenizer
            or BM25Tokenizer()
        )

        self._document_tokens: list[
            list[str]
        ] = []

        self._term_frequencies: list[
            Counter[str]
        ] = []

        self._document_frequencies: dict[
            str,
            int,
        ] = defaultdict(int)

        self._document_lengths: list[
            int
        ] = []

        self._average_document_length: float = 0.0

        self._build()

    # =========================================================
    # Index construction
    # =========================================================

    def _build(
        self,
    ) -> None:

        total_length = 0

        for chunk in self.chunks:

            searchable_text = self._searchable_text(
                chunk
            )

            tokens = self.tokenizer.tokenize(
                searchable_text
            )

            frequencies = Counter(
                tokens
            )

            self._document_tokens.append(
                tokens
            )

            self._term_frequencies.append(
                frequencies
            )

            document_length = len(
                tokens
            )

            self._document_lengths.append(
                document_length
            )

            total_length += (
                document_length
            )

            for term in frequencies:

                self._document_frequencies[
                    term
                ] += 1

        if self.chunks:

            self._average_document_length = (
                total_length
                / len(self.chunks)
            )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        language: str | None = None,
    ) -> list[LexicalSearchResult]:

        if top_k <= 0:
            return []

        query_tokens = (
            self.tokenizer.tokenize(
                query
            )
        )

        if not query_tokens:
            return []

        unique_query_terms = list(
            dict.fromkeys(
                query_tokens
            )
        )

        results: list[
            LexicalSearchResult
        ] = []

        for document_index, chunk in enumerate(
            self.chunks
        ):

            if (
                language is not None
                and chunk.language != language
            ):
                continue

            score = 0.0

            matched_terms: list[
                str
            ] = []

            for term in unique_query_terms:

                term_score = self._term_score(
                    term=term,
                    document_index=(
                        document_index
                    ),
                )

                if term_score > 0:

                    score += term_score

                    matched_terms.append(
                        term
                    )

            if score <= 0:
                continue

            results.append(
                LexicalSearchResult(
                    chunk_id=chunk.chunk_id,

                    file_id=chunk.file_id,

                    relative_path=(
                        chunk.relative_path
                    ),

                    symbol_id=chunk.symbol_id,

                    qualified_name=(
                        chunk.qualified_name
                    ),

                    chunk_type=(
                        chunk.chunk_type
                    ),

                    language=chunk.language,

                    start_line=(
                        chunk.start_line
                    ),

                    end_line=(
                        chunk.end_line
                    ),

                    score=score,

                    matched_terms=(
                        matched_terms
                    ),

                    excerpt=self._excerpt(
                        chunk.text,
                        matched_terms,
                    ),
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.relative_path.casefold(),
                result.start_line,
                result.chunk_id,
            )
        )

        return results[:top_k]

    # =========================================================
    # BM25 scoring
    # =========================================================

    def _term_score(
        self,
        term: str,
        document_index: int,
    ) -> float:

        frequencies = (
            self._term_frequencies[
                document_index
            ]
        )

        frequency = frequencies.get(
            term,
            0,
        )

        if frequency == 0:
            return 0.0

        total_documents = len(
            self.chunks
        )

        document_frequency = (
            self._document_frequencies.get(
                term,
                0,
            )
        )

        if (
            total_documents == 0
            or document_frequency == 0
        ):
            return 0.0

        idf = math.log(
            1.0
            + (
                (
                    total_documents
                    - document_frequency
                    + 0.5
                )
                /
                (
                    document_frequency
                    + 0.5
                )
            )
        )

        document_length = (
            self._document_lengths[
                document_index
            ]
        )

        average_length = (
            self._average_document_length
            or 1.0
        )

        denominator = (
            frequency
            + self.k1
            * (
                1.0
                - self.b
                + self.b
                * (
                    document_length
                    / average_length
                )
            )
        )

        numerator = (
            frequency
            * (
                self.k1
                + 1.0
            )
        )

        return (
            idf
            * (
                numerator
                / denominator
            )
        )

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _searchable_text(
        chunk: CodeChunk,
    ) -> str:

        metadata_parts = [
            chunk.relative_path,
            chunk.language,
            chunk.chunk_type.value,
        ]

        if chunk.qualified_name:

            metadata_parts.append(
                chunk.qualified_name
            )

        metadata = " ".join(
            metadata_parts
        )

        return (
            f"{metadata}\n"
            f"{chunk.text}"
        )

    @staticmethod
    def _excerpt(
        text: str,
        matched_terms: list[str],
        *,
        max_length: int = 240,
    ) -> str:

        collapsed = " ".join(
            text.split()
        )

        if not collapsed:
            return ""

        lowered = (
            collapsed.casefold()
        )

        positions = [
            lowered.find(term)
            for term in matched_terms
            if lowered.find(term) >= 0
        ]

        if positions:

            start = max(
                min(positions) - 80,
                0,
            )

        else:

            start = 0

        excerpt = collapsed[
            start:
            start + max_length
        ]

        if start > 0:

            excerpt = (
                "..."
                + excerpt
            )

        if (
            start + max_length
            < len(collapsed)
        ):

            excerpt += "..."

        return excerpt

    def __len__(
        self,
    ) -> int:

        return len(
            self.chunks
        )