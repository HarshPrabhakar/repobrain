"""Run a fixed regression benchmark; --live also uses installed local AI models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repobrain.application import RepositoryRuntimeBuilder
from repobrain.evaluation.providers import OfflineEmbeddingProvider, OfflineLLMProvider


def evaluate(live=False):
    base = Path(__file__).resolve().parents[1]
    root = base / 'evaluation' / 'fixture'
    cases = json.loads((base / 'evaluation' / 'questions.json').read_text())
    rows, latencies = [], []
    with tempfile.TemporaryDirectory(prefix='repobrain-eval-') as cache:
        provider = OfflineEmbeddingProvider()
        kwargs = {} if live else {'embedding_provider_factory': lambda: provider}
        builder = RepositoryRuntimeBuilder(index_dir=Path(cache), **kwargs)
        if not live:
            builder._build_ollama_provider = lambda: OfflineLLMProvider()
        started = time.perf_counter()
        runtime = builder.build(root)
        cold_seconds = time.perf_counter() - started
        encoded = provider.documents_encoded
        runtime.close()
        started = time.perf_counter()
        runtime = builder.build(root)
        warm_seconds = time.perf_counter() - started
        warm_reused = None if live else provider.documents_encoded == encoded
        try:
            for case in cases['retrieval']:
                started = time.perf_counter()
                result = runtime._agent_runner(case['query'])
                items = result.evidence.items if result.evidence else []
                paths = [i.relative_path for i in items]
                rank = next((i + 1 for i, p in enumerate(paths) if p == case['expected_path']), None)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                rows.append(dict(kind='retrieval', query=case['query'], success=rank is not None,
                                 reciprocal_rank=1 / rank if rank else 0, paths=paths, seconds=elapsed))
            for kind in ('graph', 'explanations'):
                for case in cases[kind]:
                    started = time.perf_counter()
                    turn = runtime.create_conversation().run(case['query'])
                    answer = turn.answer
                    facts = turn.agent_turn.agent_result.graph_facts
                    if kind == 'graph':
                        callers = 'Who calls' in case['query']
                        names = {(f.source_qualified_name if callers else f.target_qualified_name).split('.')[-1] for f in facts}
                        correct = names == set(case['expected_names']) and answer.model_name == 'deterministic-graph'
                    else:
                        correct = any(c.relative_path == case['expected_path'] for c in answer.citations)
                    valid = bool(answer.citations) and not answer.invalid_citation_ids
                    for c in answer.citations:
                        path = root / (c.relative_path or '__missing__')
                        valid = valid and path.is_file() and c.start_line is not None and c.end_line is not None
                        if valid:
                            valid = 1 <= c.start_line <= c.end_line <= len(path.read_text().splitlines()) and bool(c.source_hash)
                    elapsed = time.perf_counter() - started
                    latencies.append(elapsed)
                    rows.append(dict(kind=kind, query=case['query'], success=correct and valid,
                                     citation_valid=bool(valid), grounded=answer.grounded,
                                     answer=answer.answer_text, seconds=elapsed))
        finally:
            runtime.close()
    retrieval = [r for r in rows if r['kind'] == 'retrieval']
    graph = [r for r in rows if r['kind'] == 'graph']
    answers = [r for r in rows if r['kind'] != 'retrieval']
    return dict(mode='live' if live else 'offline-regression',
                scope='Small fixture regression; offline models do not measure real semantic or LLM quality. Citation validity checks locations, not entailment.',
                cases=rows, cold_index_seconds=cold_seconds, warm_index_seconds=warm_seconds,
                warm_embedding_reuse=warm_reused,
                retrieval_hit_rate=sum(r['success'] for r in retrieval) / len(retrieval),
                retrieval_mrr=sum(r['reciprocal_rank'] for r in retrieval) / len(retrieval),
                graph_accuracy=sum(r['success'] for r in graph) / len(graph),
                citation_location_validity=sum(r['citation_valid'] for r in answers) / len(answers),
                latency_p50_seconds=statistics.median(latencies),
                latency_p95_seconds=sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)],
                passed=all(r['success'] for r in rows) and warm_reused is not False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Use CodeRankEmbed and configured Ollama model')
    parser.add_argument('--output', type=Path, default=Path('evaluation-report.json'))
    args = parser.parse_args()
    result = evaluate(args.live)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
