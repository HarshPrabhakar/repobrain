from __future__ import annotations

import re

from repobrain.models.agent import (
    AgentIntent,
)


class DeterministicAgentRouter:
    """
    Phase 7 deterministic intent router.

    This intentionally performs no LLM inference.
    """

    _CALLER_PATTERNS = (
        "who calls",
        "what calls",
        "callers of",
        "caller of",
        "called by",
    )

    _CALLEE_PATTERNS = (
        "what does",
        "callees of",
        "callee of",
        "calls which",
    )

    _DOCUMENTATION_PATTERNS = (
        "readme",
        "documentation",
        "docs",
        "project purpose",
        "project trying to build",
        "what is this project",
        "what does this project",
    )

    _IMPLEMENTATION_PATTERNS = (
        "which function",
        "where does",
        "where is",
        "how does",
        "implementation",
        "implements",
        "compute",
        "computes",
        "prevent",
        "prevents",
        "handles",
        "processes",
    )

    _DEFINITION_PATTERNS = (
        "define",
        "definition of",
        "class named",
        "function named",
        "method named",
        "symbol named",
    )

    @classmethod
    def classify(
        cls,
        query: str,
    ) -> AgentIntent:

        normalized = (
            query.strip().casefold()
        )

        if not normalized:
            return AgentIntent.GENERAL

        if any(
            pattern in normalized
            for pattern
            in cls._CALLER_PATTERNS
        ):
            return AgentIntent.CALLERS

        if (
            "what does" in normalized
            and " call" in normalized
        ):
            return AgentIntent.CALLEES

        if any(
            pattern in normalized
            for pattern
            in cls._CALLEE_PATTERNS[1:]
        ):
            return AgentIntent.CALLEES

        if any(
            pattern in normalized
            for pattern
            in cls._DOCUMENTATION_PATTERNS
        ):
            return AgentIntent.DOCUMENTATION

        if any(
            pattern in normalized
            for pattern
            in cls._DEFINITION_PATTERNS
        ):
            return AgentIntent.DEFINITION

        if any(
            pattern in normalized
            for pattern
            in cls._IMPLEMENTATION_PATTERNS
        ):
            return AgentIntent.IMPLEMENTATION

        return AgentIntent.GENERAL

    @staticmethod
    def extract_symbol_reference(
        query: str,
    ) -> str | None:
        """
        Extract the most likely explicit symbol reference.

        This is deliberately conservative.

        Examples:

            who calls _calculate_sha256?
                -> _calculate_sha256

            what does RepositoryScanner.scan call?
                -> RepositoryScanner.scan
        """

        cleaned = (
            query
            .strip()
            .rstrip(
                "?.!,;:"
            )
        )

        patterns = (
            r"(?i)\bwho\s+calls\s+([A-Za-z_][A-Za-z0-9_.]*)",
            r"(?i)\bwhat\s+calls\s+([A-Za-z_][A-Za-z0-9_.]*)",
            r"(?i)\bcallers?\s+of\s+([A-Za-z_][A-Za-z0-9_.]*)",
            r"(?i)\bwhat\s+does\s+([A-Za-z_][A-Za-z0-9_.]*)\s+call\b",
            r"(?i)\bcallees?\s+of\s+([A-Za-z_][A-Za-z0-9_.]*)",
        )

        for pattern in patterns:

            match = re.search(
                pattern,
                cleaned,
            )

            if match:
                return match.group(1)

        # Explicit dotted symbol.
        dotted = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b",
            cleaned,
        )

        if dotted:
            return dotted[-1]

        # Explicit private-ish identifier is often a strong symbol hint.
        private_names = re.findall(
            r"\b_[A-Za-z_][A-Za-z0-9_]*\b",
            cleaned,
        )

        if private_names:
            return private_names[-1]

        return None