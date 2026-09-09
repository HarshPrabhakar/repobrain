from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable

from repobrain.models.hybrid import (
    GraphRetrievalEvidence,
    HybridSearchResult,
    RetrievalChannel,
    RetrievalEvidence,
)

from repobrain.models.retrieval import (
    ChunkType,
)


DEFAULT_RRF_K = 60.0


@dataclass
class _HybridAccumulator:
    """
    Internal mutable accumulator used only during fusion.
    """

    result_key: str

    symbol_id: str | None = None
    chunk_id: str | None = None

    file_id: str | None = None
    relative_path: str | None = None

    qualified_name: str | None = None

    chunk_type: ChunkType | None = None

    start_line: int | None = None
    end_line: int | None = None

    excerpt: str | None = None

    fused_score: float = 0.0

    channels: set[
        RetrievalChannel
    ] = field(
        default_factory=set,
    )

    evidence: list[
        RetrievalEvidence
    ] = field(
        default_factory=list,
    )


class ReciprocalRankFusion:
    """
    RepoBrain Phase 5 rank fusion.

    RRF avoids directly comparing incompatible raw scores from:

        symbol search
        BM25
        semantic similarity
        graph expansion

    Formula:

        contribution =
            weight / (k + rank)

    Raw retriever scores are preserved only as evidence.
    """

    def __init__(
        self,
        *,
        k: float = DEFAULT_RRF_K,
        symbol_weight: float = 1.0,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
        graph_weight: float = 0.8,
    ) -> None:

        if k <= 0:
            raise ValueError(
                "RRF k must be > 0."
            )

        self.k = float(
            k
        )

        self.weights = {
            RetrievalChannel.SYMBOL: (
                float(symbol_weight)
            ),
            RetrievalChannel.BM25: (
                float(bm25_weight)
            ),
            RetrievalChannel.SEMANTIC: (
                float(semantic_weight)
            ),
            RetrievalChannel.GRAPH: (
                float(graph_weight)
            ),
        }

    # =========================================================
    # Public fusion API
    # =========================================================

    def fuse(
        self,
        *,
        symbol_results: Iterable[
            Any
        ] = (),
        lexical_results: Iterable[
            Any
        ] = (),
        semantic_results: Iterable[
            Any
        ] = (),
        graph_results: Iterable[
            GraphRetrievalEvidence
        ] = (),
        top_k: int = 10,
    ) -> list[
        HybridSearchResult
    ]:

        if top_k <= 0:
            return []

        accumulators: dict[
            str,
            _HybridAccumulator,
        ] = {}

        self._consume_results(
            accumulators=accumulators,
            results=symbol_results,
            channel=(
                RetrievalChannel.SYMBOL
            ),
        )

        self._consume_results(
            accumulators=accumulators,
            results=lexical_results,
            channel=(
                RetrievalChannel.BM25
            ),
        )

        self._consume_results(
            accumulators=accumulators,
            results=semantic_results,
            channel=(
                RetrievalChannel.SEMANTIC
            ),
        )

        self._consume_graph_results(
            accumulators=accumulators,
            results=graph_results,
        )

        final_results = [
            self._finalize(
                accumulator
            )
            for accumulator
            in accumulators.values()
        ]

        final_results.sort(
            key=lambda result: (
                -result.fused_score,
                -len(
                    result.channels
                ),
                (
                    result.qualified_name
                    or result.relative_path
                    or result.result_key
                ).casefold(),
                result.result_key,
            )
        )

        return final_results[
            :top_k
        ]

    # =========================================================
    # Standard retrieval channels
    # =========================================================

    def _consume_results(
        self,
        *,
        accumulators: dict[
            str,
            _HybridAccumulator,
        ],
        results: Iterable[
            Any
        ],
        channel: RetrievalChannel,
    ) -> None:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            key = (
                self._result_key(
                    result
                )
            )

            accumulator = (
                accumulators.get(
                    key
                )
            )

            if accumulator is None:

                accumulator = (
                    self._accumulator_from_result(
                        key=key,
                        result=result,
                    )
                )

                accumulators[
                    key
                ] = accumulator

            else:

                self._merge_metadata(
                    accumulator=accumulator,
                    result=result,
                )

            contribution = (
                self._rrf_contribution(
                    channel=channel,
                    rank=rank,
                )
            )

            raw_score = (
                self._raw_score(
                    result,
                    channel=channel,
                )
            )

            accumulator.fused_score += (
                contribution
            )

            accumulator.channels.add(
                channel
            )

            accumulator.evidence.append(
                RetrievalEvidence(
                    channel=channel,
                    rank=rank,
                    raw_score=raw_score,
                    contribution=(
                        contribution
                    ),
                )
            )

    # =========================================================
    # Graph evidence
    # =========================================================

    def _consume_graph_results(
        self,
        *,
        accumulators: dict[
            str,
            _HybridAccumulator,
        ],
        results: Iterable[
            GraphRetrievalEvidence
        ],
    ) -> None:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            key = (
                f"symbol:{result.symbol_id}"
            )

            accumulator = (
                accumulators.get(
                    key
                )
            )

            if accumulator is None:

                accumulator = (
                    _HybridAccumulator(
                        result_key=key,
                        symbol_id=(
                            result.symbol_id
                        ),
                        file_id=(
                            result.file_id
                        ),
                        qualified_name=(
                            result.qualified_name
                        ),
                    )
                )

                accumulators[
                    key
                ] = accumulator

            base_contribution = (
                self._rrf_contribution(
                    channel=(
                        RetrievalChannel.GRAPH
                    ),
                    rank=rank,
                )
            )

            # Graph evidence weakens with distance.
            contribution = (
                base_contribution
                / float(
                    result.hop_distance
                )
            )

            accumulator.fused_score += (
                contribution
            )

            accumulator.channels.add(
                RetrievalChannel.GRAPH
            )

            accumulator.evidence.append(
                RetrievalEvidence(
                    channel=(
                        RetrievalChannel.GRAPH
                    ),

                    rank=rank,

                    contribution=(
                        contribution
                    ),

                    relationship_type=(
                        result.relationship_type
                    ),

                    hop_distance=(
                        result.hop_distance
                    ),

                    seed_symbol_id=(
                        result.seed_symbol_id
                    ),

                    details=(
                        f"{result.direction} "
                        f"{result.relationship_type.value}"
                    ),
                )
            )

    # =========================================================
    # RRF math
    # =========================================================

    def _rrf_contribution(
        self,
        *,
        channel: RetrievalChannel,
        rank: int,
    ) -> float:

        weight = self.weights[
            channel
        ]

        return (
            weight
            / (
                self.k
                + float(rank)
            )
        )

    # =========================================================
    # Result identity
    # =========================================================

    @staticmethod
    def _result_key(
        result: Any,
    ) -> str:

        symbol_id = getattr(
            result,
            "symbol_id",
            None,
        )

        if symbol_id:
            return (
                f"symbol:{symbol_id}"
            )

        chunk_id = getattr(
            result,
            "chunk_id",
            None,
        )

        if chunk_id:
            return (
                f"chunk:{chunk_id}"
            )

        relative_path = getattr(
            result,
            "relative_path",
            None,
        )

        start_line = getattr(
            result,
            "start_line",
            None,
        )

        end_line = getattr(
            result,
            "end_line",
            None,
        )

        if relative_path:

            return (
                "location:"
                f"{relative_path}:"
                f"{start_line}:"
                f"{end_line}"
            )

        qualified_name = getattr(
            result,
            "qualified_name",
            None,
        )

        if qualified_name:

            return (
                "qualified:"
                f"{qualified_name}"
            )

        raise ValueError(
            "Unable to create hybrid result key."
        )

    # =========================================================
    # Metadata
    # =========================================================

    @staticmethod
    def _accumulator_from_result(
        *,
        key: str,
        result: Any,
    ) -> _HybridAccumulator:

        return _HybridAccumulator(
            result_key=key,

            symbol_id=getattr(
                result,
                "symbol_id",
                None,
            ),

            chunk_id=getattr(
                result,
                "chunk_id",
                None,
            ),

            file_id=getattr(
                result,
                "file_id",
                None,
            ),

            relative_path=getattr(
                result,
                "relative_path",
                None,
            ),

            qualified_name=getattr(
                result,
                "qualified_name",
                None,
            ),

            chunk_type=getattr(
                result,
                "chunk_type",
                None,
            ),

            start_line=getattr(
                result,
                "start_line",
                None,
            ),

            end_line=getattr(
                result,
                "end_line",
                None,
            ),

            excerpt=getattr(
                result,
                "excerpt",
                None,
            ),
        )

    @staticmethod
    def _merge_metadata(
        *,
        accumulator: _HybridAccumulator,
        result: Any,
    ) -> None:

        attributes = (
            "symbol_id",
            "chunk_id",
            "file_id",
            "relative_path",
            "qualified_name",
            "chunk_type",
            "start_line",
            "end_line",
            "excerpt",
        )

        for attribute in attributes:

            current_value = getattr(
                accumulator,
                attribute,
            )

            if current_value is not None:
                continue

            incoming_value = getattr(
                result,
                attribute,
                None,
            )

            if incoming_value is not None:

                setattr(
                    accumulator,
                    attribute,
                    incoming_value,
                )

    # =========================================================
    # Score extraction
    # =========================================================

    @staticmethod
    def _raw_score(
        result: Any,
        *,
        channel: RetrievalChannel,
    ) -> float | None:

        if (
            channel
            == RetrievalChannel.SEMANTIC
        ):

            ranking_score = getattr(
                result,
                "ranking_score",
                None,
            )

            if ranking_score is not None:

                return float(
                    ranking_score
                )

        score = getattr(
            result,
            "score",
            None,
        )

        if score is None:
            return None

        return float(
            score
        )

    # =========================================================
    # Final conversion
    # =========================================================

    @staticmethod
    def _finalize(
        accumulator: _HybridAccumulator,
    ) -> HybridSearchResult:

        channels = sorted(
            accumulator.channels,
            key=lambda value: (
                value.value
            ),
        )

        evidence = sorted(
            accumulator.evidence,
            key=lambda item: (
                item.channel.value,
                item.rank,
            ),
        )

        return HybridSearchResult(
            result_key=(
                accumulator.result_key
            ),

            symbol_id=(
                accumulator.symbol_id
            ),

            chunk_id=(
                accumulator.chunk_id
            ),

            file_id=(
                accumulator.file_id
            ),

            relative_path=(
                accumulator.relative_path
            ),

            qualified_name=(
                accumulator.qualified_name
            ),

            chunk_type=(
                accumulator.chunk_type
            ),

            start_line=(
                accumulator.start_line
            ),

            end_line=(
                accumulator.end_line
            ),

            excerpt=(
                accumulator.excerpt
            ),

            fused_score=(
                accumulator.fused_score
            ),

            channels=channels,

            evidence=evidence,
        )