from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from repobrain.embeddings.base import (
    EmbeddingProvider,
)


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_EMBEDDING_MODEL = "nomic-ai/CodeRankEmbed"

FALLBACK_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

QUERY_PREFIX = (
    "Represent this query for searching relevant code: "
)

DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 4


# ============================================================================
# Compatibility helpers
# ============================================================================


def _get_embedding_dimension_from_model(
    model: SentenceTransformer,
) -> int:
    """
    Determine embedding dimension across supported
    SentenceTransformers versions.
    """

    # Newer API
    get_dimension = getattr(
        model,
        "get_embedding_dimension",
        None,
    )

    if callable(get_dimension):
        dimension = get_dimension()

        if dimension is not None:
            return int(dimension)

    # SentenceTransformers 5.1.x API
    get_sentence_dimension = getattr(
        model,
        "get_sentence_embedding_dimension",
        None,
    )

    if callable(get_sentence_dimension):
        dimension = (
            get_sentence_dimension()
        )

        if dimension is not None:
            return int(dimension)

    raise RuntimeError(
        "Unable to determine embedding dimension "
        "from SentenceTransformer model."
    )


def _normalize_device(
    device: str | None,
) -> str:
    """
    Resolve the execution device.
    """

    if (
        device is None
        or device.strip().lower() == "auto"
    ):
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    normalized = (
        device.strip().lower()
    )

    if (
        normalized.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        return "cpu"

    return normalized


# ============================================================================
# Provider
# ============================================================================


class SentenceTransformerEmbeddingProvider(
    EmbeddingProvider
):
    """
    SentenceTransformer embedding provider used by RepoBrain.

    Default model:
        nomic-ai/CodeRankEmbed

    Behavior:
        - query-specific CodeRankEmbed instruction
        - raw document/code embeddings
        - normalized vectors
        - bounded sequence length
        - GPU execution when available
        - conservative batching
        - CUDA OOM retry
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_sequence_length: int = (
            DEFAULT_MAX_SEQUENCE_LENGTH
        ),
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        query_prefix: str = QUERY_PREFIX,
    ) -> None:

        normalized_model_name = (
            model_name.strip()
        )

        if not normalized_model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be > 0."
            )

        # ------------------------------------------------------------
        # Store configuration using private backing fields.
        #
        # model_name and dimension are abstract properties defined
        # by EmbeddingProvider, so do NOT assign directly to
        # self.model_name or self.dimension.
        # ------------------------------------------------------------

        self._model_name = (
            normalized_model_name
        )

        self._device = (
            _normalize_device(
                device
            )
        )

        self._batch_size = int(
            batch_size
        )

        self._max_sequence_length = int(
            max_sequence_length
        )

        self._normalize_embeddings = bool(
            normalize_embeddings
        )

        self._trust_remote_code = bool(
            trust_remote_code
        )

        self._query_prefix = (
            query_prefix
        )

        # ------------------------------------------------------------
        # Load SentenceTransformer
        # ------------------------------------------------------------

        self._model = (
            SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=(
                    self._trust_remote_code
                ),
            )
        )

        # ------------------------------------------------------------
        # Limit sequence length
        # ------------------------------------------------------------

        if hasattr(
            self._model,
            "max_seq_length",
        ):
            self._model.max_seq_length = (
                self._max_sequence_length
            )

        # ------------------------------------------------------------
        # Resolve embedding dimension once
        # ------------------------------------------------------------

        self._dimension = (
            _get_embedding_dimension_from_model(
                self._model
            )
        )

    # ========================================================================
    # Required EmbeddingProvider properties
    # ========================================================================

    @property
    def model_name(
        self,
    ) -> str:
        """
        Name of the underlying embedding model.
        """

        return self._model_name

    @property
    def dimension(
        self,
    ) -> int:
        """
        Embedding vector dimension.
        """

        return self._dimension

    # ========================================================================
    # Additional provider metadata
    # ========================================================================

    @property
    def device(
        self,
    ) -> str:
        return self._device

    @property
    def batch_size(
        self,
    ) -> int:
        return self._batch_size

    @property
    def max_sequence_length(
        self,
    ) -> int:
        return self._max_sequence_length

    @property
    def normalize_embeddings(
        self,
    ) -> bool:
        return self._normalize_embeddings

    # ========================================================================
    # Compatibility method
    # ========================================================================

    def get_embedding_dimension(
        self,
    ) -> int:
        """
        Compatibility method used by RepoBrain components.
        """

        return self._dimension

    # ========================================================================
    # Query embedding
    # ========================================================================

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        prepared_query = (
            self._prepare_query(
                normalized_query
            )
        )

        embeddings = self._encode(
            [
                prepared_query,
            ]
        )

        return embeddings[0]

    # ========================================================================
    # Multiple query embeddings
    # ========================================================================

    def embed_queries(
        self,
        queries: Sequence[str],
    ) -> np.ndarray:

        prepared: list[str] = []

        for query in queries:
            normalized = (
                str(query).strip()
            )

            if not normalized:
                raise ValueError(
                    "queries must not contain "
                    "empty strings."
                )

            prepared.append(
                self._prepare_query(
                    normalized
                )
            )

        if not prepared:
            return np.empty(
                (
                    0,
                    self._dimension,
                ),
                dtype=np.float32,
            )

        return self._encode(
            prepared
        )

    # ========================================================================
    # Document embedding
    # ========================================================================

    def embed_document(
        self,
        document: str,
    ) -> np.ndarray:

        embeddings = (
            self.embed_documents(
                [
                    document,
                ]
            )
        )

        return embeddings[0]

    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:

        prepared = [
            str(document)
            for document
            in documents
        ]

        if not prepared:
            return np.empty(
                (
                    0,
                    self._dimension,
                ),
                dtype=np.float32,
            )

        return self._encode(
            prepared
        )

    # ========================================================================
    # Query preparation
    # ========================================================================

    def _prepare_query(
        self,
        query: str,
    ) -> str:

        if (
            self._model_name
            == DEFAULT_EMBEDDING_MODEL
        ):
            return (
                f"{self._query_prefix}"
                f"{query}"
            )

        return query

    # ========================================================================
    # Encoding
    # ========================================================================

    def _encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (
                    0,
                    self._dimension,
                ),
                dtype=np.float32,
            )

        try:
            embeddings = (
                self._model.encode(
                    list(texts),
                    batch_size=(
                        self._batch_size
                    ),
                    convert_to_numpy=True,
                    normalize_embeddings=(
                        self._normalize_embeddings
                    ),
                    show_progress_bar=False,
                )
            )

        except torch.cuda.OutOfMemoryError:

            if not self._device.startswith(
                "cuda"
            ):
                raise

            torch.cuda.empty_cache()

            fallback_batch_size = max(
                1,
                self._batch_size // 2,
            )

            embeddings = (
                self._model.encode(
                    list(texts),
                    batch_size=(
                        fallback_batch_size
                    ),
                    convert_to_numpy=True,
                    normalize_embeddings=(
                        self._normalize_embeddings
                    ),
                    show_progress_bar=False,
                )
            )

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim == 1:
            array = array.reshape(
                1,
                -1,
            )

        if array.ndim != 2:
            raise RuntimeError(
                "Unexpected embedding shape: "
                f"{array.shape}"
            )

        if (
            array.shape[1]
            != self._dimension
        ):
            raise RuntimeError(
                "Embedding dimension mismatch. "
                f"Expected {self._dimension}, "
                f"received {array.shape[1]}."
            )

        return array