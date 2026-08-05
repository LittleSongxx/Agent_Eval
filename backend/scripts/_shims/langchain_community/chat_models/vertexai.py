"""Placeholder for the VertexAI chat model removed from langchain-community.

See scripts/_shims/langchain_community/__init__.py for why this exists.
ragas references ``ChatVertexAI`` only for its LLM factory registry and never
instantiates it when a custom wrapper is injected.
"""


class ChatVertexAI:  # pragma: no cover - never instantiated in the comparison flow
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "shim ChatVertexAI: use an OpenAI-compatible wrapper instead"
        )
