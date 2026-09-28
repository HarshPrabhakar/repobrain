# RepoBrain: service and indexing upgrade

This is a complete source bundle based on your uploaded repobrain2.zip.
Copy its contents into `D:\repobrain` after backing up your existing source.
Keep your existing `venv` and Ollama installation. No model weights are included.

## Run on Windows

```powershell
cd D:\repobrain
.\venv\Scripts\activate
python -m pip install -e ".[test]"
python -m pytest -q
python scripts\evaluate.py --output evaluation-report.json
python scripts\run_api.py
```

Open http://127.0.0.1:8000/. Open a repository, watch indexing progress,
start a session, and ask a question. Click a citation to see the saved source
with line numbers. Swagger documentation is at http://127.0.0.1:8000/docs.

For real-model evaluation (requires CodeRankEmbed and Ollama):

```powershell
ollama list
python scripts\evaluate.py --live --output evaluation-report-live.json
```

This command may download the embedding model if it is not cached. Ensure
`qwen2.5-coder:14b` is installed and Ollama is running. The local test environment
used offline providers; Qwen/GPU results must be checked on your machine.

## What changed

- Existing synchronous repository/session/query endpoints remain available.
- `POST /indexing/jobs` accepts `{"repository_root":"D:/Projects/example"}`
  and returns HTTP 202 with a job ID. Poll `GET /indexing/jobs/{job_id}` for
  queued/running/completed/failed status, progress, and final runtime details.
- The dashboard uses the job endpoint. One indexing worker limits concurrent
  model builds; up to eight jobs can be active/queued. Job history is in memory
  and keeps up to 100 records. Jobs and conversations do not survive restart.
- SQLite saves raw per-file analysis, resolved snapshots, chunks, semantic
  vectors, serialized FAISS indexes, source text, and content fingerprints.
- An unchanged restart restores analysis/chunks/FAISS. The embedding provider
  still loads for subsequent queries, but document embeddings are not regenerated.
- After content changes, Python parsing is reused for unchanged files. Resolution
  runs across the repository again to handle imports and cross-file references.
  Semantic documents are rebuilt; only changed document texts are embedded.
  BM25 and graph structures are rebuilt in memory. This is conservative incremental
  indexing, not a fully in-place graph/BM25 mutation algorithm.
- Additions, modifications, deletions, parsed/reused files, and embedding hits/misses
  appear in final job details. Reopening an already-loaded unchanged runtime may
  skip the build and therefore provide no new build counters.
- Updates occur when reopening/indexing or when an operation checks the existing
  fingerprint. There is no background filesystem watcher. Old sessions remain
  protected by the existing stale-session behavior; start a new session after updates.
- Citations now include file path, start/end lines, SHA-256 source hash, index
  fingerprint, and a source URL. Graph citations link to the calling/source symbol.
  `/source` serves saved text only, never arbitrary client-supplied filesystem paths.
  Old citation URLs keep working after source edits and reindexing.
- SQLite publishes a complete snapshot transactionally. A failed build leaves the
  previous saved snapshot intact; changed-during-build repositories are rejected.
- ML imports are deferred until provider construction, letting offline tests run
  without loading PyTorch or Sentence Transformers.
- FastAPI/Uvicorn/HTTPX are declared in packaging; requirements use pyproject.toml
  as the single dependency source. Existing embedding dependency pins are preserved.

## Configuration

Set these in PowerShell before starting the server:

```powershell
$env:REPOBRAIN_INDEX_DIR = "D:\repobrain-indexes"
$env:REPOBRAIN_OLLAMA_HOST = "http://localhost:11434"
$env:REPOBRAIN_OLLAMA_MODEL = "qwen2.5-coder:14b"
$env:REPOBRAIN_EMBEDDING_DEVICE = "auto"
```

The index directory defaults to `~/.cache/repobrain`. It must be outside the
repository being indexed. Device choices include `auto`, `cpu`, and `cuda`.
Set `REPOBRAIN_EMBEDDING_CACHE_REVISION` to a new value when changing model weights
under the same model identifier. Model name, dimension, sequence length, normalization,
and semantic-document schema also participate in vector cache identity.

Historical source generations and old cache entries are retained for citation
stability. There is no automatic garbage collection yet. To reclaim all cached
versions, stop the service and remove the corresponding SQLite file; the next open
will rebuild it and old citation URLs will no longer resolve.

## Docker

```powershell
cd D:\repobrain
Copy-Item .env.example .env
# Edit REPOBRAIN_REPOSITORIES_DIR in .env to the folder containing your repositories.
docker compose up --build -d
docker compose logs -f repobrain
```

Open http://127.0.0.1:8000/ and enter a container path, e.g.
`/repositories/my-project`, rather than `D:\Projects\my-project`.
The mount is read-only; indexes/model caches live in the `repobrain-data` volume.
`docker compose down` preserves that volume; `docker compose down -v` removes it.

This initial container uses CPU embeddings and your host's Ollama service for LLM
inference. Make Ollama reachable from Docker at `host.docker.internal:11434`.
If it only listens on host loopback, configure Ollama's `OLLAMA_HOST` to a reachable
host interface and restart it; restrict access to your local Docker setup.
The API binds to localhost on the host and has no authentication; this is a local
single-user deployment, not a public server. Use one Uvicorn worker because
runtime/session/job state is process-local.

Docker build and live inference were not executed in the implementation environment.
The image installs dependencies at build time; the dependency ranges are not a
fully reproducible lockfile. First model use needs network access unless pre-cached.

References used for the container setup:
- https://fastapi.tiangolo.com/deployment/docker/
- https://docs.docker.com/compose/how-tos/networking/
- https://docs.docker.com/reference/compose-file/volumes/

## Evaluation

`python scripts/evaluate.py` runs a fixed 12-case fixture: six retrieval questions,
four caller/callee questions, and two citation-backed explanation questions.
It outputs retrieval hit rate/MRR, exact graph-answer accuracy, citation location
validity, cold/warm build time, query p50/p95, and an overall pass/fail exit code.

The default mode uses token-hash embeddings and a fixed citation-producing LLM
stub. Its results measure regression behavior, not semantic model quality or
natural-language answer correctness. `--live` exercises the real embedding/LLM
stack, but even then location-valid citations do not prove claim entailment.
Expand the golden dataset across real repositories before making quality claims.

The GitHub Actions workflow runs the test suite and offline evaluation and uploads
JUnit/evaluation reports. It will run after these files are pushed to your repository;
no remote workflow was started here.
