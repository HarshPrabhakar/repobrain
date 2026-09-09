from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath

from repobrain.models.retrieval import (
    ChunkType,
    SemanticSearchResult,
)


class SemanticQueryIntent(str, Enum):
    """
    Deterministic semantic-query intent.

    Phase 3C.3 intentionally keeps this small.

    This is not meant to replace the future RepoBrain agent.
    It only helps semantic retrieval rank candidate evidence.
    """

    IMPLEMENTATION = "IMPLEMENTATION"
    DOCUMENTATION = "DOCUMENTATION"
    DEFINITION = "DEFINITION"
    GENERAL = "GENERAL"


class SemanticSourceKind(str, Enum):
    """
    Broad source role used during semantic reranking.
    """

    PRODUCTION = "PRODUCTION"
    TEST = "TEST"
    DOCUMENTATION = "DOCUMENTATION"
    CONFIG = "CONFIG"
    SCRIPT = "SCRIPT"
    OTHER = "OTHER"


class SemanticIntentClassifier:
    """
    Lightweight deterministic intent classifier.

    No LLM is involved.
    """

    _DOCUMENTATION_PATTERNS = (
        "what is this project",
        "what does this project",
        "what is the project",
        "what does the project",
        "trying to build",
        "project purpose",
        "project overview",
        "overview of",
        "readme",
        "documentation",
        "docs",
        "how to use",
        "how do i use",
        "how to install",
        "installation",
        "setup instructions",
    )

    _DEFINITION_PATTERNS = (
        "where is defined",
        "where is it defined",
        "where is this defined",
        "find the definition",
        "definition of",
        "define ",
        "which class defines",
        "which function defines",
    )

    _IMPLEMENTATION_PATTERNS = (
        "which function",
        "which method",
        "where does",
        "where is",
        "how does",
        "how is",
        "implemented",
        "implementation",
        "computes",
        "compute ",
        "calculated",
        "calculate ",
        "handles",
        "handle ",
        "prevents",
        "prevent ",
        "validates",
        "validate ",
        "creates",
        "create ",
        "builds",
        "build ",
        "generates",
        "generate ",
        "reads ",
        "writes ",
        "loads ",
        "saves ",
    )

    @classmethod
    def classify(
        cls,
        query: str,
    ) -> SemanticQueryIntent:

        normalized = cls._normalize(
            query
        )

        if not normalized:
            return (
                SemanticQueryIntent.GENERAL
            )

        if any(
            pattern in normalized
            for pattern
            in cls._DOCUMENTATION_PATTERNS
        ):
            return (
                SemanticQueryIntent.DOCUMENTATION
            )

        if any(
            pattern in normalized
            for pattern
            in cls._DEFINITION_PATTERNS
        ):
            return (
                SemanticQueryIntent.DEFINITION
            )

        if any(
            pattern in normalized
            for pattern
            in cls._IMPLEMENTATION_PATTERNS
        ):
            return (
                SemanticQueryIntent.IMPLEMENTATION
            )

        return (
            SemanticQueryIntent.GENERAL
        )

    @staticmethod
    def _normalize(
        query: str,
    ) -> str:

        value = query.casefold()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()


class SemanticSourceClassifier:
    """
    Classify a repository path into a broad source role.

    This classification is deterministic and based only on
    the repository-relative path.
    """

    _CONFIG_FILENAMES = {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "mypy.ini",
        "pytest.ini",
        ".editorconfig",
        "package.json",
        "tsconfig.json",
    }

    _CONFIG_EXTENSIONS = {
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".conf",
    }

    _DOCUMENT_EXTENSIONS = {
        ".md",
        ".rst",
        ".adoc",
        ".txt",
    }

    @classmethod
    def classify(
        cls,
        relative_path: str,
    ) -> SemanticSourceKind:

        normalized = (
            relative_path
            .replace("\\", "/")
        )

        path = PurePosixPath(
            normalized
        )

        parts = {
            part.casefold()
            for part in path.parts
        }

        file_name = (
            path.name.casefold()
        )

        suffix = (
            path.suffix.casefold()
        )

        # -----------------------------------------------------
        # Tests
        # -----------------------------------------------------

        if (
            "tests" in parts
            or "test" in parts
            or file_name.startswith(
                "test_"
            )
            or file_name.endswith(
                "_test.py"
            )
        ):
            return (
                SemanticSourceKind.TEST
            )

        # -----------------------------------------------------
        # Documentation
        # -----------------------------------------------------

        if (
            "docs" in parts
            or "documentation" in parts
            or file_name.startswith(
                "readme"
            )
            or file_name.startswith(
                "changelog"
            )
            or suffix
            in cls._DOCUMENT_EXTENSIONS
        ):
            return (
                SemanticSourceKind.DOCUMENTATION
            )

        # -----------------------------------------------------
        # Configuration
        # -----------------------------------------------------

        if (
            file_name
            in cls._CONFIG_FILENAMES
            or suffix
            in cls._CONFIG_EXTENSIONS
        ):
            return (
                SemanticSourceKind.CONFIG
            )

        # -----------------------------------------------------
        # Scripts
        # -----------------------------------------------------

        if (
            "scripts" in parts
            or "tools" in parts
            or file_name
            in {
                "run.py",
                "main.py",
            }
        ):
            return (
                SemanticSourceKind.SCRIPT
            )

        # -----------------------------------------------------
        # Production RepoBrain package
        # -----------------------------------------------------

        if (
            path.parts
            and path.parts[0].casefold()
            == "repobrain"
        ):
            return (
                SemanticSourceKind.PRODUCTION
            )

        return (
            SemanticSourceKind.OTHER
        )


class SemanticRankingPolicy:
    """
    Phase 3C.3 deterministic semantic reranker.

    The raw FAISS semantic score remains untouched.

    ranking_score =
        semantic similarity
        + intent-aware chunk adjustment
        + intent-aware source adjustment

    These adjustments are ranking signals, not confidence
    probabilities.
    """

    def rerank(
        self,
        *,
        query: str,
        results: list[
            SemanticSearchResult
        ],
        top_k: int,
    ) -> list[
        SemanticSearchResult
    ]:

        if top_k <= 0:
            return []

        intent = (
            SemanticIntentClassifier
            .classify(
                query
            )
        )

        reranked: list[
            SemanticSearchResult
        ] = []

        for result in results:

            source_kind = (
                SemanticSourceClassifier
                .classify(
                    result.relative_path
                )
            )

            adjustment = 0.0

            reasons: list[str] = []

            # -------------------------------------------------
            # Chunk role
            # -------------------------------------------------

            (
                chunk_adjustment,
                chunk_reason,
            ) = self._chunk_adjustment(
                intent=intent,
                chunk_type=(
                    result.chunk_type
                ),
            )

            adjustment += (
                chunk_adjustment
            )

            if chunk_reason:
                reasons.append(
                    chunk_reason
                )

            # -------------------------------------------------
            # Source role
            # -------------------------------------------------

            (
                source_adjustment,
                source_reason,
            ) = self._source_adjustment(
                intent=intent,
                source_kind=(
                    source_kind
                ),
            )

            adjustment += (
                source_adjustment
            )

            if source_reason:
                reasons.append(
                    source_reason
                )

            ranking_score = (
                float(result.score)
                + adjustment
            )

            reranked.append(
                result.model_copy(
                    update={
                        "ranking_score": (
                            ranking_score
                        ),
                        "query_intent": (
                            intent.value
                        ),
                        "source_kind": (
                            source_kind.value
                        ),
                        "ranking_reasons": (
                            reasons
                        ),
                    }
                )
            )

        reranked.sort(
            key=lambda item: (
                -(
                    item.ranking_score
                    if item.ranking_score
                    is not None
                    else item.score
                ),
                -item.score,
                item.relative_path.casefold(),
                item.start_line,
                item.chunk_id,
            )
        )

        return reranked[
            :top_k
        ]

    # =========================================================
    # Chunk weighting
    # =========================================================

    @staticmethod
    def _chunk_adjustment(
        *,
        intent: SemanticQueryIntent,
        chunk_type: ChunkType,
    ) -> tuple[
        float,
        str | None,
    ]:

        if (
            intent
            == SemanticQueryIntent.IMPLEMENTATION
        ):

            weights = {
                ChunkType.METHOD: (
                    0.10,
                    "implementation intent: method boost",
                ),

                ChunkType.FUNCTION: (
                    0.09,
                    "implementation intent: function boost",
                ),

                ChunkType.CLASS: (
                    0.03,
                    "implementation intent: class boost",
                ),

                ChunkType.MODULE: (
                    -0.03,
                    "implementation intent: module penalty",
                ),

                ChunkType.FILE: (
                    -0.05,
                    "implementation intent: generic file penalty",
                ),
            }

            return weights.get(
                chunk_type,
                (0.0, None),
            )

        if (
            intent
            == SemanticQueryIntent.DOCUMENTATION
        ):

            weights = {
                ChunkType.FILE: (
                    0.08,
                    "documentation intent: file boost",
                ),

                ChunkType.MODULE: (
                    -0.01,
                    "documentation intent: module penalty",
                ),

                ChunkType.CLASS: (
                    -0.02,
                    "documentation intent: class penalty",
                ),

                ChunkType.FUNCTION: (
                    -0.03,
                    "documentation intent: function penalty",
                ),

                ChunkType.METHOD: (
                    -0.03,
                    "documentation intent: method penalty",
                ),
            }

            return weights.get(
                chunk_type,
                (0.0, None),
            )

        if (
            intent
            == SemanticQueryIntent.DEFINITION
        ):

            weights = {
                ChunkType.CLASS: (
                    0.08,
                    "definition intent: class boost",
                ),

                ChunkType.FUNCTION: (
                    0.08,
                    "definition intent: function boost",
                ),

                ChunkType.METHOD: (
                    0.08,
                    "definition intent: method boost",
                ),

                ChunkType.MODULE: (
                    -0.02,
                    "definition intent: module penalty",
                ),
            }

            return weights.get(
                chunk_type,
                (0.0, None),
            )

        return (
            0.0,
            None,
        )

    # =========================================================
    # Source weighting
    # =========================================================

    @staticmethod
    def _source_adjustment(
        *,
        intent: SemanticQueryIntent,
        source_kind: SemanticSourceKind,
    ) -> tuple[
        float,
        str | None,
    ]:

        if (
            intent
            == SemanticQueryIntent.IMPLEMENTATION
        ):

            weights = {
                SemanticSourceKind.PRODUCTION: (
                    0.08,
                    "implementation intent: production source boost",
                ),

                SemanticSourceKind.SCRIPT: (
                    0.01,
                    "implementation intent: script source boost",
                ),

                SemanticSourceKind.TEST: (
                    -0.12,
                    "implementation intent: test source penalty",
                ),

                SemanticSourceKind.DOCUMENTATION: (
                    -0.12,
                    "implementation intent: documentation penalty",
                ),

                SemanticSourceKind.CONFIG: (
                    -0.10,
                    "implementation intent: config penalty",
                ),

                SemanticSourceKind.OTHER: (
                    0.0,
                    None,
                ),
            }

            return weights[
                source_kind
            ]

        if (
            intent
            == SemanticQueryIntent.DOCUMENTATION
        ):

            weights = {
                SemanticSourceKind.DOCUMENTATION: (
                    0.15,
                    "documentation intent: documentation source boost",
                ),

                SemanticSourceKind.CONFIG: (
                    0.06,
                    "documentation intent: config source boost",
                ),

                SemanticSourceKind.PRODUCTION: (
                    -0.02,
                    "documentation intent: production source penalty",
                ),

                SemanticSourceKind.TEST: (
                    -0.08,
                    "documentation intent: test source penalty",
                ),

                SemanticSourceKind.SCRIPT: (
                    -0.02,
                    "documentation intent: script source penalty",
                ),

                SemanticSourceKind.OTHER: (
                    0.0,
                    None,
                ),
            }

            return weights[
                source_kind
            ]

        if (
            intent
            == SemanticQueryIntent.DEFINITION
        ):

            if (
                source_kind
                == SemanticSourceKind.TEST
            ):
                return (
                    -0.06,
                    "definition intent: test source penalty",
                )

            if (
                source_kind
                == SemanticSourceKind.PRODUCTION
            ):
                return (
                    0.04,
                    "definition intent: production source boost",
                )

        # General semantic queries receive only a very small
        # test penalty. We do not want to aggressively distort
        # genuine semantic similarity for open-ended questions.
        if (
            intent
            == SemanticQueryIntent.GENERAL
            and source_kind
            == SemanticSourceKind.TEST
        ):
            return (
                -0.03,
                "general intent: small test source penalty",
            )

        return (
            0.0,
            None,
        )