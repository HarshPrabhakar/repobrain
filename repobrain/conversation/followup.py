from __future__ import annotations

import re
from dataclasses import dataclass

from repobrain.models.conversation import (
    FollowUpReferenceKind,
    FollowUpResolution,
    InvestigationState,
    RelationshipFocusType,
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

    Supported focus classes:

    - current symbol
    - previous symbol
    - one unambiguous caller
    - one unambiguous callee

    The resolver never:

    - calls an LLM
    - performs repository retrieval
    - performs graph traversal
    - guesses among multiple relationship targets
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
    # Relationship references
    # ==================================================================

    _CALLER_REFERENCE_TEXT = (
        "the caller's",
        "that caller's",
        "this caller's",
        "the caller",
        "that caller",
        "this caller",
    )

    _CALLEE_REFERENCE_TEXT = (
        "the callee's",
        "that callee's",
        "this callee's",
        "the callee",
        "that callee",
        "this callee",
    )

    # ==================================================================
    # Bare this / that validation
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
        # No conversational reference
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
        # Current primary focus
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
        # Previous primary focus
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

        # --------------------------------------------------------------
        # Caller relationship focus
        # --------------------------------------------------------------

        elif (
            pattern.kind
            == FollowUpReferenceKind.CALLER_FOCUS
        ):

            if (
                state.relationship_focus_type
                == RelationshipFocusType.CALLER
            ):

                symbol_id = (
                    state.relationship_symbol_id
                )

                qualified_name = (
                    state.relationship_qualified_name
                )

            else:

                symbol_id = None

                qualified_name = None

        # --------------------------------------------------------------
        # Callee relationship focus
        # --------------------------------------------------------------

        elif (
            pattern.kind
            == FollowUpReferenceKind.CALLEE_FOCUS
        ):

            if (
                state.relationship_focus_type
                == RelationshipFocusType.CALLEE
            ):

                symbol_id = (
                    state.relationship_symbol_id
                )

                qualified_name = (
                    state.relationship_qualified_name
                )

            else:

                symbol_id = None

                qualified_name = None

        else:

            symbol_id = None

            qualified_name = None

        # --------------------------------------------------------------
        # Query rewriting requires deterministic qualified name
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

        replacement = (
            qualified_name
        )

        # --------------------------------------------------------------
        # Possessive relation references
        #
        # "that caller's implementation"
        #
        # becomes:
        #
        # "pkg.module.function implementation"
        # --------------------------------------------------------------

        if (
            reference_text
            .lower()
            .endswith("'s")
        ):

            replacement = (
                qualified_name
            )

        resolved_query = (
            self._replace_reference(
                query=(
                    normalized_query
                ),
                match=match,
                replacement=(
                    replacement
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
        # Relationship phrases first.
        #
        # This is important because:
        #
        #     "that caller"
        #
        # must not be consumed by the shorter "that".
        # --------------------------------------------------------------

        for phrase in (
            cls._CALLER_REFERENCE_TEXT
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
                        .CALLER_FOCUS
                    ),
                    phrase=(
                        phrase
                    ),
                )
            )

        for phrase in (
            cls._CALLEE_REFERENCE_TEXT
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
                        .CALLEE_FOCUS
                    ),
                    phrase=(
                        phrase
                    ),
                )
            )

        # --------------------------------------------------------------
        # Previous focus before generic current focus
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
                    phrase=(
                        phrase
                    ),
                )
            )

        # --------------------------------------------------------------
        # Current focus
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
                    phrase=(
                        phrase
                    ),
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
        Compile deterministic reference phrase.

        The final boundary intentionally supports possessive
        expressions such as "that caller's".
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
        Return earliest valid conversational reference.

        If several begin at the same position, longest wins.
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
        Determine whether bare this/that is acting as a
        conversational pronoun rather than a determiner.
        """

        remainder = (
            query[
                match.end():
            ]
            .lstrip()
        )

        if not remainder:
            return True

        if (
            remainder[0]
            in ".,?!;:"
        ):
            return True

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