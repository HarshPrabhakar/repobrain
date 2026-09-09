from __future__ import annotations

import numpy as np
import torch

from sentence_transformers import (
    SentenceTransformer,
)

from repobrain.embeddings.base import (
    EmbeddingProvider,
)


DEFAULT_EMBEDDING_MODEL = (
    "nomic-ai/CodeRankEmbed"
)

FALLBACK_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

QUERY_PREFIX = (
    "Represent this query for searching relevant code: "
)


DEFAULT_MAX_SEQUENCE_LENGTH = 512

DEFAULT_BATCH_SIZE = 4


class SentenceTransformerEmbeddingProvider(
    EmbeddingProvider
):
    """
    Local Sentence Transformers embedding provider.

    Phase 3C.1 defaults to a code-specialized Jina model.

    Important:
    The Jina model supports a long context window, but allowing
    repository chunks to use the full context can require huge
    attention tensors and exhaust GPU VRAM.

    RepoBrain therefore applies a conservative semantic-search
    sequence-length limit.

    Embeddings are L2-normalized so FAISS IndexFlatIP behaves
    like cosine similarity.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        trust_remote_code: bool | None = None,
    ) -> None:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if max_seq_length <= 0:
            raise ValueError(
                "max_seq_length must be greater than zero."
            )

        self._model_name = model_name

        self.batch_size = batch_size

        self.max_seq_length = (
            max_seq_length
        )

        # -----------------------------------------------------
        # Device selection
        # -----------------------------------------------------

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        # -----------------------------------------------------
        # Remote-code policy
        # -----------------------------------------------------

        if trust_remote_code is None:

            trust_remote_code = (
                model_name
                == DEFAULT_EMBEDDING_MODEL
            )

        self.trust_remote_code = (
            trust_remote_code
        )

        # -----------------------------------------------------
        # Load embedding model
        # -----------------------------------------------------

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=(
                trust_remote_code
            ),
        )

        # -----------------------------------------------------
        # IMPORTANT VRAM SAFETY LIMIT
        # -----------------------------------------------------

        self.model.max_seq_length = (
            self.max_seq_length
        )

        # -----------------------------------------------------
        # Embedding dimension
        # -----------------------------------------------------

        dimension = (
            self.model
            .get_embedding_dimension()
        )

        if dimension is None:

            raise RuntimeError(
                "Embedding model did not report "
                "an embedding dimension."
            )

        self._dimension = int(
            dimension
        )

    # =========================================================
    # Metadata
    # =========================================================

    @property
    def dimension(
        self,
    ) -> int:

        return self._dimension

    @property
    def model_name(
        self,
    ) -> str:

        return self._model_name

    # =========================================================
    # Document embeddings
    # =========================================================

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

        # Clear stale allocations before a large embedding pass.
        self._clear_cuda_cache()

        try:

            embeddings = (
                self.model.encode(
                    texts,
                    batch_size=(
                        self.batch_size
                    ),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )

        except torch.OutOfMemoryError as exc:

            self._clear_cuda_cache()

            raise RuntimeError(
                "CUDA ran out of memory while "
                "embedding repository chunks. "
                "Try reducing batch_size or "
                "max_seq_length. "
                f"Current batch_size="
                f"{self.batch_size}, "
                f"max_seq_length="
                f"{self.max_seq_length}."
            ) from exc

        finally:

            self._clear_cuda_cache()

        return self._ensure_float32(
            embeddings
        )

    # =========================================================
    # Query embedding
    # =========================================================

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:

        query = text.strip()

        if not query:

            return np.zeros(
                self.dimension,
                dtype=np.float32,
            )

        if (
            self.model_name
            == DEFAULT_EMBEDDING_MODEL
        ):

            query = (
                QUERY_PREFIX
                + query
            )

        self._clear_cuda_cache()

        try:

            embedding = (
                self.model.encode(
                    [query],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )[0]
            )

        except torch.OutOfMemoryError as exc:

            self._clear_cuda_cache()

            raise RuntimeError(
                "CUDA ran out of memory while "
                "embedding the query."
            ) from exc

        finally:

            self._clear_cuda_cache()

        return self._ensure_float32(
            embedding
        )

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _ensure_float32(
        array: np.ndarray,
    ) -> np.ndarray:

        return np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    def _clear_cuda_cache(
        self,
    ) -> None:
        """
        Release unused cached CUDA blocks.

        This does not delete live model tensors.
        """

        if (
            self.device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        ):

            torch.cuda.empty_cache()