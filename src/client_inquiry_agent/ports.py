"""The outside services the agent depends on, as interfaces.

The real implementations live in gemini_client.py and brain_client.py; tests pass fakes,
so no test touches the network, Gemini or the Second Brain.
"""

from typing import Protocol


class LLMError(Exception):
    """A Gemini call failed (network, quota, timeout, empty reply...)."""


class LLM(Protocol):
    model: str

    def generate(self, prompt: str) -> str:
        """Return the model's reply text, or raise LLMError."""
        ...
