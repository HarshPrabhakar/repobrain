# Validation and changed files

Validated on Python 3.12 in the execution environment.
- Full suite: 395 passed.
- Fixed offline benchmark: 12/12 cases passed.
- Editable packaging install (--no-deps): passed.
- Dashboard JavaScript syntax: passed.
- Compose and GitHub Actions YAML parsing: passed.
- Docker build/start: not run; Docker unavailable.
- Live Ollama/CodeRankEmbed and Windows/GPU validation: not run.

See UPGRADE.md for Windows commands, deployment, limitations and evaluation interpretation.

| Change | File |
|---|---|
| Added | `.dockerignore` |
| Added | `.env.example` |
| Added | `.github/workflows/evaluate.yml` |
| Added | `Dockerfile` |
| Added | `UPGRADE.md` |
| Added | `compose.yaml` |
| Added | `evaluation/fixture/payments.py` |
| Added | `evaluation/fixture/readme.md` |
| Added | `evaluation/fixture/storage.py` |
| Added | `evaluation/questions.json` |
| Updated | `pyproject.toml` |
| Added | `repobrain/answering/versioned.py` |
| Updated | `repobrain/api/app.py` |
| Added | `repobrain/api/indexing.py` |
| Updated | `repobrain/api/ui.py` |
| Updated | `repobrain/application/builder.py` |
| Added | `repobrain/application/progress.py` |
| Updated | `repobrain/application/transport.py` |
| Updated | `repobrain/embeddings/sentence_transformer.py` |
| Added | `repobrain/evaluation/__init__.py` |
| Added | `repobrain/evaluation/providers.py` |
| Updated | `repobrain/models/answer.py` |
| Updated | `repobrain/models/application.py` |
| Added | `repobrain/persistence/__init__.py` |
| Added | `repobrain/persistence/store.py` |
| Updated | `repobrain/retrieval/semantic.py` |
| Updated | `requirements.txt` |
| Added | `scripts/evaluate.py` |
| Updated | `tests/test_api_repository_chat.py` |
| Added | `tests/test_persistent_indexing.py` |
