from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from repobrain.evidence.budgeting import (
    EvidenceBudget,
    EvidenceBudgeter,
)

from repobrain.graph import (
    RepositoryKnowledgeGraph,
)

from repobrain.models.evidence import (
    EvidenceBundle,
    EvidenceGraphRelation,
    EvidenceItem,
    EvidenceKind,
)

from repobrain.models.hybrid import (
    HybridSearchResult,
)


DEFAULT_MAX_GRAPH_RELATIONS = 5


class EvidenceAssembler:
    """
    Convert hybrid retrieval results into bounded repository evidence.

    Responsibilities:

    - recover exact chunk/source text
    - preserve retrieval provenance
    - attach graph context
    - remove duplicate evidence
    - avoid overlapping repeated source spans
    - apply deterministic context limits

    No LLM is used here.
    """

    def __init__(
        self,
        *,
        chunks: Iterable[Any],
        graph: (
            RepositoryKnowledgeGraph
            | None
        ) = None,
        budget: (
            EvidenceBudget
            | None
        ) = None,
        max_graph_relations: int = (
            DEFAULT_MAX_GRAPH_RELATIONS
        ),
    ) -> None:

        if max_graph_relations < 0:
            raise ValueError(
                "max_graph_relations must be >= 0."
            )

        self.graph = graph

        self.max_graph_relations = (
            max_graph_relations
        )

        self.budgeter = (
            EvidenceBudgeter(
                budget
            )
        )

        self._chunks_by_id: dict[
            str,
            Any,
        ] = {}

        self._chunks_by_symbol_id: dict[
            str,
            list[Any],
        ] = {}

        for chunk in chunks:

            chunk_id = getattr(
                chunk,
                "chunk_id",
                None,
            )

            if chunk_id:
                self._chunks_by_id[
                    chunk_id
                ] = chunk

            symbol_id = getattr(
                chunk,
                "symbol_id",
                None,
            )

            if symbol_id:

                self._chunks_by_symbol_id.setdefault(
                    symbol_id,
                    [],
                ).append(
                    chunk
                )

    # =====================================================================
    # Public API
    # =====================================================================

    def assemble(
        self,
        *,
        query: str,
        results: Iterable[
            HybridSearchResult
        ],
    ) -> EvidenceBundle:

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        candidates: list[
            EvidenceItem
        ] = []

        for result in results:

            item = (
                self._item_from_result(
                    result
                )
            )

            if item is None:
                continue

            candidates.append(
                item
            )

        candidates = (
            self._deduplicate(
                candidates
            )
        )

        candidates = (
            self._remove_overlapping_spans(
                candidates
            )
        )

        selected, omitted = (
            self.budgeter.apply(
                candidates
            )
        )

        total_characters = sum(
            item.character_count
            for item in selected
        )

        estimated_tokens = sum(
            item.estimated_tokens
            for item in selected
        )

        return EvidenceBundle(
            query=normalized_query,
            items=selected,
            total_characters=(
                total_characters
            ),
            estimated_tokens=(
                estimated_tokens
            ),
            omitted_items=(
                omitted
            ),
            max_items=(
                self.budgeter
                .budget
                .max_items
            ),
            max_characters=(
                self.budgeter
                .budget
                .max_characters
            ),
        )

    # =====================================================================
    # Result conversion
    # =====================================================================

    def _item_from_result(
        self,
        result: HybridSearchResult,
    ) -> EvidenceItem | None:

        chunk = (
            self._find_chunk(
                result
            )
        )

        source_text = (
            self._source_text(
                result=result,
                chunk=chunk,
            )
        )

        if not source_text.strip():
            return None

        relative_path = (
            result.relative_path
            or self._get(
                chunk,
                "relative_path",
            )
        )

        start_line = (
            result.start_line
            if result.start_line
            is not None
            else self._get(
                chunk,
                "start_line",
            )
        )

        end_line = (
            result.end_line
            if result.end_line
            is not None
            else self._get(
                chunk,
                "end_line",
            )
        )

        symbol_id = (
            result.symbol_id
            or self._get(
                chunk,
                "symbol_id",
            )
        )

        qualified_name = (
            result.qualified_name
            or self._get(
                chunk,
                "qualified_name",
            )
        )

        graph_context = (
            self._graph_context(
                symbol_id
            )
            if symbol_id
            else []
        )

        evidence_id = (
            self._create_evidence_id(
                result_key=(
                    result.result_key
                ),
                relative_path=(
                    relative_path
                ),
                start_line=(
                    start_line
                ),
                end_line=(
                    end_line
                ),
            )
        )

        return EvidenceItem(
            evidence_id=(
                evidence_id
            ),
            result_key=(
                result.result_key
            ),
            file_id=(
                result.file_id
                or self._get(
                    chunk,
                    "file_id",
                )
            ),
            relative_path=(
                relative_path
            ),
            symbol_id=(
                symbol_id
            ),
            qualified_name=(
                qualified_name
            ),
            start_line=(
                start_line
            ),
            end_line=(
                end_line
            ),
            source_text=(
                source_text
            ),
            evidence_kind=(
                self._classify_evidence(
                    relative_path=(
                        relative_path
                    ),
                    symbol_id=(
                        symbol_id
                    ),
                )
            ),
            fused_score=(
                result.fused_score
            ),
            retrieval_channels=list(
                result.channels
            ),
            graph_context=(
                graph_context
            ),
            character_count=len(
                source_text
            ),
            estimated_tokens=(
                self.budgeter
                .estimate_tokens(
                    source_text
                )
            ),
        )

    # =====================================================================
    # Chunk recovery
    # =====================================================================

    def _find_chunk(
        self,
        result: HybridSearchResult,
    ) -> Any | None:

        if result.chunk_id:

            chunk = (
                self._chunks_by_id.get(
                    result.chunk_id
                )
            )

            if chunk is not None:
                return chunk

        if result.symbol_id:

            chunks = (
                self._chunks_by_symbol_id.get(
                    result.symbol_id,
                    [],
                )
            )

            if chunks:

                return sorted(
                    chunks,
                    key=lambda chunk: (
                        self._get(
                            chunk,
                            "start_line",
                        )
                        or 0,
                        self._get(
                            chunk,
                            "chunk_id",
                        )
                        or "",
                    ),
                )[0]

        return None

    # =====================================================================
    # Source extraction
    # =====================================================================

    @staticmethod
    def _source_text(
        *,
        result: HybridSearchResult,
        chunk: Any | None,
    ) -> str:

        if chunk is not None:

            # RepoBrain versions have used slightly different
            # names while the chunk model evolved.
            for attribute in (
                "source",
                "source_text",
                "content",
                "text",
            ):

                value = getattr(
                    chunk,
                    attribute,
                    None,
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.strip()
                ):
                    return value

        if result.excerpt:
            return result.excerpt

        return ""

    # =====================================================================
    # Graph context
    # =====================================================================

    def _graph_context(
        self,
        symbol_id: str,
    ) -> list[
        EvidenceGraphRelation
    ]:

        if (
            self.graph is None
            or self.max_graph_relations == 0
        ):
            return []

        relations: list[
            EvidenceGraphRelation
        ] = []

        neighbors = (
            self.graph.neighbors(
                symbol_id,
                direction="both",
            )
        )

        for neighbor in neighbors:

            relations.append(
                EvidenceGraphRelation(
                    relationship_type=(
                        neighbor
                        .relationship
                        .relationship_type
                    ),
                    direction=(
                        neighbor.direction
                    ),
                    related_symbol_id=(
                        neighbor
                        .symbol
                        .symbol_id
                    ),
                    related_qualified_name=(
                        neighbor
                        .symbol
                        .qualified_name
                    ),
                    line_number=(
                        neighbor
                        .relationship
                        .line_number
                    ),
                )
            )

            if (
                len(relations)
                >= self.max_graph_relations
            ):
                break

        return relations

    # =====================================================================
    # Deduplication
    # =====================================================================

    @staticmethod
    def _deduplicate(
        items: list[
            EvidenceItem
        ],
    ) -> list[
        EvidenceItem
    ]:

        seen: set[str] = set()

        output: list[
            EvidenceItem
        ] = []

        for item in items:

            key = (
                EvidenceAssembler
                ._deduplication_key(
                    item
                )
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            output.append(
                item
            )

        return output

    @staticmethod
    def _deduplication_key(
        item: EvidenceItem,
    ) -> str:

        if item.symbol_id:
            return (
                f"symbol:{item.symbol_id}"
            )

        if (
            item.relative_path
            and item.start_line is not None
            and item.end_line is not None
        ):
            return (
                "location:"
                f"{item.relative_path}:"
                f"{item.start_line}:"
                f"{item.end_line}"
            )

        digest = hashlib.sha256(
            item.source_text.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"content:{digest}"
        )

    # =====================================================================
    # Overlap handling
    # =====================================================================

    @staticmethod
    def _remove_overlapping_spans(
        items: list[
            EvidenceItem
        ],
    ) -> list[
        EvidenceItem
    ]:
        """
        Avoid sending near-identical overlapping source regions.

        Higher-ranked results win because incoming order already
        reflects hybrid ranking.
        """

        accepted: list[
            EvidenceItem
        ] = []

        for item in items:

            if (
                item.relative_path is None
                or item.start_line is None
                or item.end_line is None
            ):
                accepted.append(
                    item
                )
                continue

            overlaps = False

            for existing in accepted:

                if (
                    existing.relative_path
                    != item.relative_path
                ):
                    continue

                if (
                    existing.start_line is None
                    or existing.end_line is None
                ):
                    continue

                if EvidenceAssembler._spans_overlap(
                    item.start_line,
                    item.end_line,
                    existing.start_line,
                    existing.end_line,
                ):
                    overlaps = True
                    break

            if not overlaps:
                accepted.append(
                    item
                )

        return accepted

    @staticmethod
    def _spans_overlap(
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> bool:

        return (
            start_a <= end_b
            and start_b <= end_a
        )

    # =====================================================================
    # Classification
    # =====================================================================

    @staticmethod
    def _classify_evidence(
        *,
        relative_path: str | None,
        symbol_id: str | None,
    ) -> EvidenceKind:

        if relative_path:

            normalized = (
                relative_path
                .replace(
                    "\\",
                    "/",
                )
                .casefold()
            )

            if (
                normalized.startswith(
                    "tests/"
                )
                or "/tests/" in normalized
                or normalized
                .split("/")[-1]
                .startswith(
                    "test_"
                )
            ):
                return (
                    EvidenceKind.TEST
                )

            if normalized.endswith(
                (
                    ".md",
                    ".rst",
                )
            ):
                return (
                    EvidenceKind.DOCUMENTATION
                )

            filename = (
                normalized
                .split("/")[-1]
            )

            if (
                filename
                in {
                    "pyproject.toml",
                    "setup.cfg",
                    "setup.py",
                    "requirements.txt",
                }
                or normalized.endswith(
                    (
                        ".yaml",
                        ".yml",
                        ".toml",
                        ".ini",
                    )
                )
            ):
                return (
                    EvidenceKind.CONFIG
                )

        if symbol_id:
            return EvidenceKind.SYMBOL

        if relative_path:
            return EvidenceKind.FILE

        return EvidenceKind.OTHER

    # =====================================================================
    # Utility
    # =====================================================================

    @staticmethod
    def _create_evidence_id(
        *,
        result_key: str,
        relative_path: str | None,
        start_line: int | None,
        end_line: int | None,
    ) -> str:

        payload = (
            f"{result_key}|"
            f"{relative_path}|"
            f"{start_line}|"
            f"{end_line}"
        )

        return hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _get(
        value: Any | None,
        attribute: str,
    ) -> Any | None:

        if value is None:
            return None

        return getattr(
            value,
            attribute,
            None,
        )