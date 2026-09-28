"""Versioned SQLite index cache. JSON models and raw vectors; no pickle loading."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

SCHEMA_VERSION = 1


class IndexStore:
    def __init__(self, directory: Path, repository_root: Path):
        directory = Path(directory).expanduser().resolve()
        if directory.is_relative_to(Path(repository_root).expanduser().resolve()):
            raise ValueError("REPOBRAIN_INDEX_DIR must be outside the indexed repository")
        directory.mkdir(parents=True, exist_ok=True)
        root = str(Path(repository_root).resolve())
        self.path = directory / (hashlib.sha256(root.encode()).hexdigest() + '.sqlite3')
        with self.connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError('Unsupported index schema; use a fresh REPOBRAIN_INDEX_DIR')
            db.execute('CREATE TABLE IF NOT EXISTS cache (kind TEXT, key TEXT, value BLOB NOT NULL, PRIMARY KEY(kind,key))')
            db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, kind, key):
        with self.connect() as db:
            row = db.execute('SELECT value FROM cache WHERE kind=? AND key=?', (kind, key)).fetchone()
        return row[0] if row else None

    def put(self, kind, key, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (kind, key, value))

    def get_json(self, kind, key):
        value = self.get(kind, key)
        return json.loads(value) if value is not None else None

    def put_json(self, kind, key, value):
        self.put(kind, key, json.dumps(value, sort_keys=True))

    def analyze(self, scan, analyzer):
        """Cache raw per-file ASTs; never reuse previously resolved relationships."""
        from repobrain.models.symbols import PythonRepositoryAnalysis
        result = PythonRepositoryAnalysis(repository_id=scan.metadata.repository_id)
        reused = parsed = 0
        for file in scan.files:
            if file.language != 'Python':
                continue
            key = f'{file.file_id}:{file.content_hash}:ast-v1'
            cached = self.get('analysis', key) if file.content_hash else None
            if cached is not None:
                part = PythonRepositoryAnalysis.model_validate_json(cached)
                reused += 1
            else:
                part = analyzer.analyze(scan.model_copy(update={'files': [file]}))
                parsed += 1
                if hashlib.sha256(Path(file.absolute_path).read_bytes()).hexdigest() != file.content_hash:
                    raise RuntimeError("Repository changed during parsing; retry indexing.")
                if not part.errors and file.content_hash:
                    self.put('analysis', key, part.model_dump_json())
            result.files_analyzed += part.files_analyzed
            result.symbols.extend(part.symbols)
            result.relationships.extend(part.relationships)
            result.errors.extend(part.errors)
        self.parse_stats = {'parsed_files': parsed, 'reused_files': reused}
        return result

    def commit_snapshot(self, scan, analysis, chunks, fingerprint):
        """Publish one complete generation atomically; readers never see half a manifest."""
        sources = []
        for file in scan.files:
            data = Path(file.absolute_path).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != file.content_hash:
                raise RuntimeError('Repository changed during indexing; retry opening it.')
            sources.append((digest, data.decode('utf-8', errors='replace')))
        previous = self.get_json('snapshot', 'current')
        previous_files = {f['relative_path']: f['content_hash'] for f in previous['scan']['files']} if previous else {}
        current_files = {f.relative_path: f.content_hash for f in scan.files}
        changes = {
            'added_files': len(current_files.keys() - previous_files.keys()),
            'deleted_files': len(previous_files.keys() - current_files.keys()),
            'modified_files': sum(current_files[p] != previous_files[p] for p in current_files.keys() & previous_files.keys()),
        }
        self.change_stats = changes
        payload = {'changes': changes, 'scan': scan.model_dump(mode='json'),
                   'analysis': analysis.model_dump(mode='json'),
                   'chunks': [c.model_dump(mode='json') for c in chunks],
                   'fingerprint': fingerprint.value,
                   'stats': getattr(self, 'parse_stats', {})}
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO cache VALUES (?,?,?)',
                           [('source', digest, text) for digest, text in sources])
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)',
                       ('snapshot', 'current', json.dumps(payload)))
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)',
                       ('generation', fingerprint.value, json.dumps(payload)))


class CachedEmbeddingProvider:
    """Reuse vectors by exact semantic document and explicit embedding configuration."""
    def __init__(self, provider, store, namespace):
        self.provider, self.store, self.namespace = provider, store, namespace
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def embed_query(self, text):
        return self.provider.embed_query(text)

    def embed_documents(self, texts):
        keys = [hashlib.sha256((self.namespace + '\0' + t).encode()).hexdigest() for t in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts):
            raw = self.store.get('vector', key)
            if raw is not None:
                vector = np.frombuffer(raw, dtype=np.float32)
                if vector.shape == (self.dimension,) and np.isfinite(vector).all():
                    vectors[key] = vector
                    self.hits += 1
                    continue
            missing[key] = text
        if missing:
            encoded = np.asarray(self.provider.embed_documents(list(missing.values())), dtype=np.float32)
            if encoded.shape != (len(missing), self.dimension) or not np.isfinite(encoded).all():
                raise ValueError('Invalid document embeddings')
            with self.store.connect() as db:
                for key, vector in zip(missing, encoded):
                    vectors[key] = vector
                    db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', ('vector', key, vector.tobytes()))
            self.misses += len(missing)
        return np.asarray([vectors[k] for k in keys], dtype=np.float32).reshape(len(texts), self.dimension)
