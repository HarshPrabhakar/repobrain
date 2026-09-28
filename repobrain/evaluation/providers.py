"""Deterministic offline providers. These do not measure embedding/LLM quality."""
import hashlib
import re
import numpy as np


class OfflineEmbeddingProvider:
    model_name = 'offline-token-hash-v1'
    dimension = 64
    device = 'cpu'
    max_sequence_length = 512
    normalize_embeddings = True

    def __init__(self):
        self.documents_encoded = 0

    def embed_query(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r'\w+', text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dimension] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        self.documents_encoded += len(texts)
        return np.asarray([self.embed_query(t) for t in texts], dtype=np.float32)


class OfflineLLMProvider:
    model_name = 'offline-citation-fixture'

    def generate(self, *, instructions, input_text):
        return 'The implementation is provided in the cited source. [E1]'
