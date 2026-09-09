from __future__ import annotations

import re
from dataclasses import dataclass

from repobrain.models.conversation import (
    FollowUpReferenceKind,
    FollowUpResolution,
    InvestigationState,
)


@dataclass(
    frozen=True,
    slots=True,
)
class _ReferencePattern:
    """
    Internal deterministic reference pattern.
    """

    expression: re.Pattern[str]

    kind: FollowUpReferenceKind

    phrase: str


class DeterministicFollowUpResolver:
    """
    Resolve bounded conversational references using RepoBrain's
    deterministic InvestigationState.

    Examples:

        "Where is that defined?"

    becomes:

        "Where is
        repobrain.ingestion.scanner.RepositoryScanner._calculate_sha256
        defined?"

    The resolver is deliberately conservative.

    It does not:

    - call an LLM
    - perform repository retrieval
    - inspect source code
    - guess ambiguous references
    """

    # ==================================================================
    # Current-focus references
    # ==================================================================

    _CURRENT_REFERENCE_TEXT = (
        "this function",
        "that function",
        "same function",
        "current function",
        "this method",
        "that method",
        "same method",
        "current method",
        "this class",
        "that class",
        "same class",
        "current class",
        "this symbol",
        "that symbol",
        "same symbol",
        "current symbol",
        "it",
        "that",
        "this",
    )

    # ==================================================================
    # Previous-focus references
    # ==================================================================

    _PREVIOUS_REFERENCE_TEXT = (
        "the previous function",
        "the previous method",
        "the previous class",
        "the previous symbol",
        "previous function",
        "previous method",
        "previous class",
        "previous symbol",
    )

    # ==================================================================
    # Relationship references intentionally unsupported for now
    # ==================================================================

    _UNSUPPORTED_RELATION_TEXT = (
        "the caller",
        "that caller",
        "this caller",
        "the callee",
        "that callee",
        "this callee",
    )

    # ==================================================================
    # Bare "this" / "that" handling
    #
    # A bare demonstrative is only considered referential when the next
    # token strongly suggests that the user is referring back to an
    # already-known subject.
    #
    # This prevents ordinary phrases such as:
    #
    #     this project
    #     this repository
    #     this architecture
    #
    # from being mistaken for conversation references.
    # ==================================================================

    _BARE_REFERENCE_ALLOWED_NEXT_WORDS = frozenset(
        {
            "call",
            "calls",
            "called",
            "defined",
            "implemented",
            "implement",
            "work",
            "works",
            "return",
            "returns",
            "returned",
            "use",
            "uses",
            "used",
            "depend",
            "depends",
            "import",
            "imports",
            "inherit",
            "inherits",
            "contain",
            "contains",
            "do",
            "does",
            "did",
            "mean",
            "means",
            "located",
        }
    )

    def __init__(
        self,
    ) -> None:

        self._patterns = (
            self._build_patterns()
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def resolve(
        self,
        query: str,
        state: InvestigationState,
    ) -> FollowUpResolution:
        """
        Resolve one query against deterministic investigation state.

        A query with no conversational reference passes through
        unchanged.

        A conversational reference that cannot safely be resolved
        returns unresolved_reference=True.
        """

        normalized_query = (
            self._normalize_query(
                query
            )
        )

        match_result = (
            self._find_reference(
                normalized_query
            )
        )

        # --------------------------------------------------------------
        # No conversational reference.
        # --------------------------------------------------------------

        if match_result is None:

            return FollowUpResolution(
                original_query=(
                    normalized_query
                ),
                resolved_query=(
                    normalized_query
                ),
            )

        pattern, match = (
            match_result
        )

        reference_text = (
            match.group(0)
        )

        # --------------------------------------------------------------
        # Relationship references are not yet represented by explicit
        # relationship-focus state.
        # --------------------------------------------------------------

        if (
            pattern.kind
            == FollowUpReferenceKind.UNSUPPORTED_RELATION
        ):

            return FollowUpResolution(
                original_query=(
                    normalized_query
                ),
                resolved_query=(
                    normalized_query
                ),
                used_context=False,
                reference_kind=(
                    FollowUpReferenceKind.UNSUPPORTED_RELATION
                ),
                reference_text=(
                    reference_text
                ),
                unresolved_reference=True,
            )

        # --------------------------------------------------------------
        # Current focus
        # --------------------------------------------------------------

        if (
            pattern.kind
            == FollowUpReferenceKind.CURRENT_FOCUS
        ):

            symbol_id = (
                state.current_symbol_id
            )

            qualified_name = (
                state.current_qualified_name
            )

        # --------------------------------------------------------------
        # Previous focus
        # --------------------------------------------------------------

        elif (
            pattern.kind
            == FollowUpReferenceKind.PREVIOUS_FOCUS
        ):

            symbol_id = (
                state.previous_symbol_id
            )

            qualified_name = (
                state.previous_qualified_name
            )

        else:

            symbol_id = None
            qualified_name = None

        # --------------------------------------------------------------
        # Query rewriting requires a qualified name.
        # --------------------------------------------------------------

        if not qualified_name:

            return FollowUpResolution(
                original_query=(
                    normalized_query
                ),
                resolved_query=(
                    normalized_query
                ),
                used_context=False,
                reference_kind=(
                    pattern.kind
                ),
                reference_text=(
                    reference_text
                ),
                unresolved_reference=True,
            )

        resolved_query = (
            self._replace_reference(
                query=(
                    normalized_query
                ),
                match=match,
                replacement=(
                    qualified_name
                ),
            )
        )

        return FollowUpResolution(
            original_query=(
                normalized_query
            ),
            resolved_query=(
                resolved_query
            ),
            used_context=True,
            reference_kind=(
                pattern.kind
            ),
            reference_text=(
                reference_text
            ),
            resolved_symbol_id=(
                symbol_id
            ),
            resolved_qualified_name=(
                qualified_name
            ),
            unresolved_reference=False,
        )

    # ==================================================================
    # Pattern construction
    # ==================================================================

    @classmethod
    def _build_patterns(
        cls,
    ) -> tuple[
        _ReferencePattern,
        ...,
    ]:

        patterns: list[
            _ReferencePattern
        ] = []

        # --------------------------------------------------------------
        # Unsupported relationships first.
        #
        # Example:
        #
        #     "that caller"
        #
        # must not be consumed by the shorter "that" pattern.
        # --------------------------------------------------------------

        for phrase in (
            cls._UNSUPPORTED_RELATION_TEXT
        ):

            patterns.append(
                _ReferencePattern(
                    expression=(
                        cls._compile_phrase(
                            phrase
                        )
                    ),
                    kind=(
                        FollowUpReferenceKind
                        .UNSUPPORTED_RELATION
                    ),
                    phrase=phrase,
                )
            )

        # --------------------------------------------------------------
        # Previous-focus phrases before generic current-focus ones.
        # --------------------------------------------------------------

        for phrase in (
            cls._PREVIOUS_REFERENCE_TEXT
        ):

            patterns.append(
                _ReferencePattern(
                    expression=(
                        cls._compile_phrase(
                            phrase
                        )
                    ),
                    kind=(
                        FollowUpReferenceKind
                        .PREVIOUS_FOCUS
                    ),
                    phrase=phrase,
                )
            )

        # --------------------------------------------------------------
        # Current-focus phrases.
        # --------------------------------------------------------------

        for phrase in (
            cls._CURRENT_REFERENCE_TEXT
        ):

            patterns.append(
                _ReferencePattern(
                    expression=(
                        cls._compile_phrase(
                            phrase
                        )
                    ),
                    kind=(
                        FollowUpReferenceKind
                        .CURRENT_FOCUS
                    ),
                    phrase=phrase,
                )
            )

        return tuple(
            patterns
        )

    @staticmethod
    def _compile_phrase(
        phrase: str,
    ) -> re.Pattern[str]:
        """
        Compile a phrase using word boundaries.

        Whitespace between phrase tokens may vary.
        """

        words = (
            phrase.split()
        )

        body = r"\s+".join(
            re.escape(
                word
            )
            for word in words
        )

        return re.compile(
            rf"(?<!\w){body}(?!\w)",
            flags=re.IGNORECASE,
        )

    # ==================================================================
    # Matching
    # ==================================================================

    def _find_reference(
        self,
        query: str,
    ) -> tuple[
        _ReferencePattern,
        re.Match[str],
    ] | None:
        """
        Return the earliest valid conversational reference.

        If several references begin at the same position, the longest
        phrase wins.
        """

        candidates: list[
            tuple[
                int,
                int,
                _ReferencePattern,
                re.Match[str],
            ]
        ] = []

        for pattern in (
            self._patterns
        ):

            match = (
                pattern.expression.search(
                    query
                )
            )

            if match is None:
                continue

            # ----------------------------------------------------------
            # Bare "this" and "that" need extra context checking.
            #
            # Examples:
            #
            #   "Where is that defined?"
            #       -> valid conversational reference
            #
            #   "What is this project?"
            #       -> ordinary English, NOT a follow-up reference
            # ----------------------------------------------------------

            if (
                pattern.phrase
                in {
                    "this",
                    "that",
                }
                and not self._bare_reference_is_valid(
                    query=query,
                    match=match,
                )
            ):

                continue

            candidates.append(
                (
                    match.start(),
                    -(
                        match.end()
                        - match.start()
                    ),
                    pattern,
                    match,
                )
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        _, _, pattern, match = (
            candidates[0]
        )

        return (
            pattern,
            match,
        )

    # ==================================================================
    # Bare-reference validation
    # ==================================================================

    @classmethod
    def _bare_reference_is_valid(
        cls,
        *,
        query: str,
        match: re.Match[str],
    ) -> bool:
        """
        Determine whether bare "this" or "that" is acting as a
        conversational pronoun rather than as a determiner.

        Examples:

            "Where is that defined?"
                -> True

            "Explain that."
                -> True

            "What is this project?"
                -> False

            "How does this system work?"
                -> False
        """

        remainder = (
            query[
                match.end():
            ]
            .lstrip()
        )

        # --------------------------------------------------------------
        # End of sentence / punctuation:
        #
        #     Explain that.
        #     Show this?
        # --------------------------------------------------------------

        if not remainder:

            return True

        if (
            remainder[0]
            in ".,?!;:"
        ):

            return True

        # --------------------------------------------------------------
        # Look at the immediate next word.
        # --------------------------------------------------------------

        next_word_match = (
            re.match(
                r"([A-Za-z_][A-Za-z0-9_]*)",
                remainder,
            )
        )

        if (
            next_word_match
            is None
        ):

            return True

        next_word = (
            next_word_match
            .group(1)
            .lower()
        )

        return (
            next_word
            in cls
            ._BARE_REFERENCE_ALLOWED_NEXT_WORDS
        )

    # ==================================================================
    # Query rewriting
    # ==================================================================

    @staticmethod
    def _replace_reference(
        *,
        query: str,
        match: re.Match[str],
        replacement: str,
    ) -> str:
        """
        Replace exactly one primary conversational reference.
        """

        rewritten = (
            query[
                :match.start()
            ]
            + replacement
            + query[
                match.end():
            ]
        )

        return (
            " ".join(
                rewritten.split()
            )
        )

    # ==================================================================
    # Query normalization
    # ==================================================================

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:
        """
        Normalize superficial whitespace.
        """

        normalized = (
            " ".join(
                query.split()
            )
        )

        if not normalized:

            raise ValueError(
                "query cannot be empty."
            )

        return normalized