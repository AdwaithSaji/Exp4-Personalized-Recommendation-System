"""Thin wrapper around the Ollama chat API that returns parsed JSON."""
import json
import re

import ollama

from .config import LLM_MODEL, LLM_TEMPERATURE


def chat(prompt, system=None, as_json=True):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        format="json" if as_json else None,
        options={"temperature": LLM_TEMPERATURE},
    )
    content = response["message"]["content"]
    return parse_json(content) if as_json else content


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


_embedder = None


def embedder():
    """Load the Sentence-Transformers model once and reuse it."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from .config import EMBEDDING_MODEL

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
