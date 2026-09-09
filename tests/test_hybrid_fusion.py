from __future__ import annotations

from types import SimpleNamespace

from repobrain.models.hybrid import (
    GraphRetrievalEvidence,
    RetrievalChannel,
)

from repobrain.models.retrieval import (
    ChunkType,
)

from repobrain.models.symbols import (
    RelationshipType,
)

from repobrain.retrieval.fusion import (
    ReciprocalRankFusion,
)


def result(
    *,
    symbol_id: str | None,
    chunk_id: str | None,
    score: float,
    name: str,
):

    return SimpleNamespace(
        symbol_id=symbol_id,
        chunk_id=chunk_id,
        file_id=(
            f"file_{name}"
        ),
        relative_path=(
            f"repobrain/{name}.py"
        ),
        qualified_name=(
            f"repobrain.{name}"
        ),
        chunk_type=(
            ChunkType.FUNCTION
        ),
        start_line=1,
        end_line=10,
        excerpt=name,
        score=score,
        ranking_score=None,
    )


def test_rrf_returns_empty_for_zero_top_k() -> None:

    fusion = (
        ReciprocalRankFusion()
    )

    assert (
        fusion.fuse(
            top_k=0
        )
        == []
    )


def test_single_channel_result() -> None:

    item = result(
        symbol_id="sym_a",
        chunk_id="chunk_a",
        score=1.0,
        name="a",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                item,
            ],
            top_k=10,
        )
    )

    assert len(fused) == 1

    assert (
        RetrievalChannel.BM25
        in fused[0].channels
    )


def test_same_symbol_is_deduplicated() -> None:

    lexical = result(
        symbol_id="sym_hash",
        chunk_id="chunk_hash",
        score=5.0,
        name="hash",
    )

    semantic = result(
        symbol_id="sym_hash",
        chunk_id="chunk_hash",
        score=0.8,
        name="hash",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                lexical,
            ],
            semantic_results=[
                semantic,
            ],
            top_k=10,
        )
    )

    assert len(fused) == 1

    assert set(
        fused[0].channels
    ) == {
        RetrievalChannel.BM25,
        RetrievalChannel.SEMANTIC,
    }


def test_agreement_increases_fused_score() -> None:

    shared_lexical = result(
        symbol_id="shared",
        chunk_id="shared_chunk",
        score=1.0,
        name="shared",
    )

    shared_semantic = result(
        symbol_id="shared",
        chunk_id="shared_chunk",
        score=1.0,
        name="shared",
    )

    lexical_only = result(
        symbol_id="single",
        chunk_id="single_chunk",
        score=100.0,
        name="single",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                lexical_only,
                shared_lexical,
            ],

            semantic_results=[
                shared_semantic,
            ],

            top_k=10,
        )
    )

    assert (
        fused[0].symbol_id
        == "shared"
    )


def test_raw_scores_are_preserved_as_evidence() -> None:

    lexical = result(
        symbol_id="sym_a",
        chunk_id="chunk_a",
        score=7.5,
        name="a",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                lexical,
            ],
            top_k=10,
        )
    )

    assert (
        fused[0]
        .evidence[0]
        .raw_score
        == 7.5
    )


def test_semantic_ranking_score_is_used_as_raw_evidence() -> None:

    semantic = result(
        symbol_id="sym_a",
        chunk_id="chunk_a",
        score=0.2,
        name="a",
    )

    semantic.ranking_score = (
        0.4
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            semantic_results=[
                semantic,
            ],
            top_k=10,
        )
    )

    assert (
        fused[0]
        .evidence[0]
        .raw_score
        == 0.4
    )


def test_graph_evidence_merges_with_existing_symbol() -> None:

    semantic = result(
        symbol_id="sym_hash",
        chunk_id="chunk_hash",
        score=0.3,
        name="hash",
    )

    graph = (
        GraphRetrievalEvidence(
            symbol_id="sym_hash",
            qualified_name=(
                "repobrain.hash"
            ),
            file_id="file_hash",
            relationship_type=(
                RelationshipType.CALLS
            ),
            hop_distance=1,
            seed_symbol_id=(
                "sym_builder"
            ),
            direction="outgoing",
        )
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            semantic_results=[
                semantic,
            ],
            graph_results=[
                graph,
            ],
            top_k=10,
        )
    )

    assert len(fused) == 1

    assert (
        RetrievalChannel.GRAPH
        in fused[0].channels
    )

    assert (
        RetrievalChannel.SEMANTIC
        in fused[0].channels
    )


def test_graph_distance_reduces_contribution() -> None:

    near = (
        GraphRetrievalEvidence(
            symbol_id="near",
            qualified_name="app.near",
            file_id="file_near",
            relationship_type=(
                RelationshipType.CALLS
            ),
            hop_distance=1,
            seed_symbol_id="seed",
            direction="outgoing",
        )
    )

    far = (
        GraphRetrievalEvidence(
            symbol_id="far",
            qualified_name="app.far",
            file_id="file_far",
            relationship_type=(
                RelationshipType.CALLS
            ),
            hop_distance=2,
            seed_symbol_id="seed",
            direction="outgoing",
        )
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            graph_results=[
                near,
                far,
            ],
            top_k=10,
        )
    )

    assert (
        fused[0].symbol_id
        == "near"
    )


def test_results_sorted_by_fused_score() -> None:

    a = result(
        symbol_id="a",
        chunk_id="a_chunk",
        score=1.0,
        name="a",
    )

    b_lexical = result(
        symbol_id="b",
        chunk_id="b_chunk",
        score=1.0,
        name="b",
    )

    b_semantic = result(
        symbol_id="b",
        chunk_id="b_chunk",
        score=1.0,
        name="b",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                a,
                b_lexical,
            ],
            semantic_results=[
                b_semantic,
            ],
            top_k=10,
        )
    )

    assert (
        fused[0].symbol_id
        == "b"
    )


def test_chunk_identity_used_when_symbol_missing() -> None:

    lexical = result(
        symbol_id=None,
        chunk_id="file_chunk",
        score=1.0,
        name="readme",
    )

    fused = (
        ReciprocalRankFusion()
        .fuse(
            lexical_results=[
                lexical,
            ],
            top_k=10,
        )
    )

    assert (
        fused[0].result_key
        == "chunk:file_chunk"
    )