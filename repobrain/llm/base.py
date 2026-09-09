from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class LLMProvider(ABC):
    """
    Provider-independent text generation interface.

    RepoBrain's answering layer depends only on this contract.
    """

    @property
    @abstractmethod
    def model_name(
        self,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        raise NotImplementedError