from pathlib import Path
import hashlib
import time

import pytest
from fastapi.testclient import TestClient

from repobrain.application import RepositoryRuntimeBuilder, RepoBrainApplicationService
from repobrain.api import create_app
from repobrain.persistence.store import IndexStore, CachedEmbeddingProvider
from repobrain.evaluation.providers import OfflineEmbeddingProvider, OfflineLLMProvider


def setup_project(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'maths.py').write_text('def add(a, b):\n    return a + b\n\ndef total():\n    return add(2, 3)\n')
    (root / 'other.py').write_text('def unrelated():\n    return 7\n')
    return root


def make_builder(tmp_path, provider=None):
    provider = provider if provider is not None else OfflineEmbeddingProvider()
    builder = RepositoryRuntimeBuilder(index_dir=tmp_path / 'indexes', embedding_provider_factory=lambda: provider)
    builder._build_ollama_provider = lambda: OfflineLLMProvider()
    return builder, provider


def ask(service, root, query):
    session = service.create_session(root)
    return service.ask(root, session.session_id, query).answer


def test_restart_reuses_ast_chunks_and_vectors(tmp_path, monkeypatch):
    root = setup_project(tmp_path)
    builder, provider = make_builder(tmp_path)
    first = builder.build(root)
    assert provider.documents_encoded > 0
    first.close()
    builder2, provider2 = make_builder(tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError('Warm open must not parse or embed documents')
    monkeypatch.setattr(provider2, 'embed_documents', fail)
    monkeypatch.setattr('repobrain.parsing.PythonRepositoryAnalyzer.analyze', fail)
    second = builder2.build(root)
    assert first.fingerprint == second.fingerprint
    assert provider2.documents_encoded == 0
    second.close()


def test_edit_add_delete_refresh_graph_and_reuse_unchanged_ast(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    service = RepoBrainApplicationService(runtime_builder=builder)
    assert 'total' in ask(service, root, 'Who calls add?').answer_text
    (root / 'maths.py').write_text('def add(a, b):\n    return a + b\n')
    (root / 'caller.py').write_text('from maths import add\n\ndef compute():\n    return add(1, 4)\n')
    answer = ask(service, root, 'Who calls add?')
    assert 'compute' in answer.answer_text
    assert 'total' not in answer.answer_text
    store = IndexStore(tmp_path / 'indexes', root)
    snapshot = store.get_json('snapshot', 'current')
    assert snapshot['stats'] == {'parsed_files': 2, 'reused_files': 1}
    (root / 'caller.py').unlink()
    answer = ask(service, root, 'Who calls add?')
    assert 'compute' not in answer.answer_text
    assert 'no resolved internal' in answer.answer_text
    service.close_all()


def test_citation_source_remains_old_version_after_edit(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    service = RepoBrainApplicationService(runtime_builder=builder)
    answer = ask(service, root, 'Who calls add?')
    citation = answer.citations[0]
    assert citation.source_hash == hashlib.sha256((root / 'maths.py').read_bytes()).hexdigest()
    with TestClient(create_app(application=service)) as client:
        source = client.get(citation.source_url)
        assert source.status_code == 200
        assert 'return add(2, 3)' in source.text
        (root / 'maths.py').write_text('def add(a, b):\n    return a - b\n')
        service.open_repository(root)
        old = client.get(citation.source_url)
        assert old.status_code == 200
        assert 'return add(2, 3)' in old.text
        params = dict(repository_root=str(root), fingerprint=citation.index_fingerprint,
                      relative_path='../secret.txt', start_line=1, end_line=1)
        assert client.get('/source', params=params).status_code == 404
        params['relative_path'] = 'maths.py'
        params['end_line'] = 9999
        assert client.get('/source', params=params).status_code == 400


def test_index_job_completes_and_exposes_progress(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    with TestClient(create_app(application=RepoBrainApplicationService(runtime_builder=builder))) as client:
        response = client.post('/indexing/jobs', json={'repository_root': str(root)})
        assert response.status_code == 202
        job_id = response.json()['job_id']
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get('/indexing/jobs/' + job_id).json()
            if job['status'] not in ('queued', 'running'):
                break
            time.sleep(0.01)
        assert job['status'] == 'completed', job
        assert job['details']['parsed_files'] == 2
        assert job['result']['fingerprint_value']
        assert client.get('/indexing/jobs/missing').status_code == 404


def test_embedding_namespace_and_document_changes_invalidate_cache(tmp_path):
    root = setup_project(tmp_path)
    store = IndexStore(tmp_path / 'indexes', root)
    provider = OfflineEmbeddingProvider()
    CachedEmbeddingProvider(provider, store, 'v1').embed_documents(['hello', 'world'])
    assert provider.documents_encoded == 2
    CachedEmbeddingProvider(provider, store, 'v1').embed_documents(['hello', 'changed'])
    assert provider.documents_encoded == 3
    CachedEmbeddingProvider(provider, store, 'v2').embed_documents(['hello'])
    assert provider.documents_encoded == 4


def test_failed_generation_does_not_replace_current_snapshot(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    builder.build(root).close()
    store = IndexStore(tmp_path / 'indexes', root)
    before = store.get_json('snapshot', 'current')
    (root / 'other.py').write_text('def changed():\n    return 8\n')
    provider = OfflineEmbeddingProvider()
    def fail(texts):
        raise RuntimeError('simulated encoder failure')
    provider.embed_documents = fail
    builder, _ = make_builder(tmp_path, provider)
    with pytest.raises(RuntimeError, match='encoder failure'):
        builder.build(root)
    assert store.get_json('snapshot', 'current') == before


def test_index_directory_cannot_be_inside_repository(tmp_path):
    root = setup_project(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        IndexStore(root / 'indexes', root)


def test_incremental_matches_clean_rebuild(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    builder.build(root).close()
    (root / 'maths.py').write_text('def add(a, b):\n    return a + b + 1\n')
    (root / 'caller.py').write_text('from maths import add\n\ndef fresh():\n    return add(2, 5)\n')
    builder.build(root).close()
    incremental = IndexStore(tmp_path / 'indexes', root).get_json('snapshot', 'current')
    clean, _ = make_builder(tmp_path)
    clean.index_dir = tmp_path / 'clean-indexes'
    clean.build(root).close()
    rebuilt = IndexStore(clean.index_dir, root).get_json('snapshot', 'current')
    assert incremental['analysis'] == rebuilt['analysis']
    assert incremental['chunks'] == rebuilt['chunks']
    assert incremental['fingerprint'] == rebuilt['fingerprint']


def test_source_citation_survives_transport(tmp_path):
    root = setup_project(tmp_path)
    builder, _ = make_builder(tmp_path)
    with TestClient(create_app(application=RepoBrainApplicationService(runtime_builder=builder))) as client:
        session = client.post('/sessions', json={'repository_root': str(root)}).json()
        response = client.post('/query/ask', json={
            'repository_root': str(root), 'session_id': session['session_id'],
            'query': 'Explain add',
        })
        assert response.status_code == 200, response.text
        citations = response.json()['citations']
        assert citations
        for citation in citations:
            assert len(citation['source_hash']) == 64
            assert len(citation['index_fingerprint']) == 64
            assert client.get(citation['source_url']).status_code == 200
