from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """
    Provider-independent embedding interface.

    Retrieval code depends on this abstraction rather
    than directly depending on Sentence Transformers.

    Implementations must return normalized float32 vectors.
    """

    @property
    @abstractmethod
    def dimension(
        self,
    ) -> int:
        """
        Embedding vector dimension.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(
        self,
    ) -> str:
        """
        Human-readable embedding model identifier.
        """

        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Embed repository documents/chunks.

        Returns:

            shape = (documents, dimension)
            dtype = float32
        """

        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Embed one user query.

        Returns:

            shape = (dimension,)
            dtype = float32
        """

        raise NotImplementedError