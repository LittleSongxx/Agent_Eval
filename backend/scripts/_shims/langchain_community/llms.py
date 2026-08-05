"""Placeholder for the VertexAI LLM removed from langchain-community.

See scripts/_shims/langchain_community/__init__.py for why this exists.
"""


class VertexAI:  # pragma: no cover - never instantiated in the comparison flow
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "shim VertexAI: use an OpenAI-compatible wrapper instead"
        )
