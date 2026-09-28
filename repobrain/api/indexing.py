"""Single-worker local indexing jobs and immutable source excerpts."""
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import RLock
from uuid import uuid4
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from repobrain.models.application import OpenRepositoryRequest
from repobrain.persistence.store import IndexStore

from repobrain.application.progress import progress_listener


class IndexingJobs:
    def __init__(self, service):
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='repobrain-index')
        self.jobs = {}
        self.lock = RLock()
        self.closed = False

    def submit(self, root):
        root = str(Path(root).expanduser().resolve())
        if not Path(root).is_dir():
            raise HTTPException(404, 'Repository directory does not exist')
        with self.lock:
            if self.closed:
                raise HTTPException(503, 'Server is shutting down')
            active = [j for j in self.jobs.values() if j['status'] in ('queued', 'running')]
            for job in active:
                if job['repository_root'] == root:
                    return dict(job)
            if len(active) >= 8:
                raise HTTPException(429, 'Indexing queue is full')
            while len(self.jobs) >= 100:
                old = next(k for k, j in self.jobs.items() if j['status'] not in ('queued', 'running'))
                del self.jobs[old]
            job = dict(job_id=uuid4().hex, repository_root=root, status='queued',
                       step=0, total=12, message='Queued', details={})
            self.jobs[job['job_id']] = job
            self.executor.submit(self._run, job['job_id'])
            return dict(job)

    def _run(self, job_id):
        def progress(step, total, message, details):
            with self.lock:
                self.jobs[job_id].update(step=step, total=total, message=message, details=details)
        with self.lock:
            job = self.jobs[job_id]
            job['status'] = 'running'
        token = progress_listener.set(progress)
        try:
            runtime = self.service.open_repository(Path(job['repository_root']))
            with self.lock:
                job.update(status='completed', step=12, message='Ready', result=runtime.model_dump(mode='json'))
        except Exception as exc:
            with self.lock:
                job.update(status='failed', message=str(exc))
        finally:
            progress_listener.reset(token)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, 'Indexing job not found')
            return dict(self.jobs[job_id])

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_indexing_router(jobs, index_dir):
    router = APIRouter()

    @router.post('/indexing/jobs', status_code=202, tags=['indexing'])
    def start_job(request: OpenRepositoryRequest):
        return jobs.submit(request.repository_root)

    @router.get('/indexing/jobs/{job_id}', tags=['indexing'])
    def job_status(job_id: str):
        return jobs.get(job_id)

    @router.get('/source', response_class=HTMLResponse, tags=['citations'])
    def source(repository_root: str, fingerprint: str, relative_path: str,
               start_line: int = Query(ge=1), end_line: int = Query(ge=1)):
        if end_line < start_line or end_line - start_line > 10000:
            raise HTTPException(400, 'Invalid source range')
        # Read only already-indexed source blobs. Never open a client-supplied file path.
        directory = Path(index_dir).expanduser().resolve()
        import hashlib
        root = str(Path(repository_root).expanduser().resolve())
        database = directory / (hashlib.sha256(root.encode()).hexdigest() + '.sqlite3')
        if not database.is_file():
            raise HTTPException(404, 'Repository has no saved index')
        store = IndexStore(directory, Path(root))
        snapshot = store.get_json('generation', fingerprint)
        if snapshot is None:
            raise HTTPException(404, 'Source generation not found')
        file = next((f for f in snapshot['scan']['files'] if f['relative_path'] == relative_path), None)
        if file is None:
            raise HTTPException(404, 'File is not in this source generation')
        text = store.get('source', file['content_hash'])
        if text is None:
            raise HTTPException(404, 'Source snapshot not found')
        lines = text.splitlines()
        if end_line > len(lines):
            raise HTTPException(400, 'Source range exceeds indexed file')
        excerpt = '\n'.join(f'{i}: {lines[i-1]}' for i in range(start_line, end_line + 1))
        return HTMLResponse('<!doctype html><meta charset="utf-8"><title>RepoBrain source</title>'
                            '<style>body{background:#101722;color:#d9e4f2;padding:24px}pre{white-space:pre-wrap}</style>'
                            f'<h3>{escape(relative_path)}:{start_line}-{end_line}</h3>'
                            f'<p>Indexed version: {escape(fingerprint)}</p><pre>{escape(excerpt)}</pre>',
                            headers={'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'",
                                     'X-Content-Type-Options': 'nosniff'})
    return router
