"""Optional loopback-only Ollama integration for aggregate teaching prompts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import requests


DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
SYSTEM_PROMPT = """You are a local classroom planning assistant. You receive aggregate classroom
signals only. Suggest three practical, respectful, low-pressure teaching actions.
Do not diagnose learners, rank students, or treat signals as ground truth.
Return JSON only in this shape:
{"suggestions":[{"title":"string","action":"string","reason":"string"}]}"""


class LocalModelError(RuntimeError):
    """Raised when local Ollama is unavailable or returns invalid output."""


@dataclass(frozen=True)
class LocalSuggestion:
    title: str
    action: str
    reason: str


class LocalTeachingAssistant:
    """Send only aggregate summaries to an Ollama endpoint bound to loopback."""

    def __init__(
        self,
        model_tag: str,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        timeout_seconds: int = 45,
        http_post: Callable[..., Any] | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("EduSense AI only supports a local loopback Ollama endpoint.")
        if not model_tag.strip():
            raise ValueError("Enter a local Ollama model tag.")
        self.model_tag = model_tag.strip()
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> list[LocalSuggestion]:
        rows = payload.get("suggestions")
        if not isinstance(rows, list) or len(rows) != 3:
            raise LocalModelError("The local model must return exactly three suggestions.")
        suggestions: list[LocalSuggestion] = []
        for row in rows:
            if not isinstance(row, dict):
                raise LocalModelError("Each suggestion must be a JSON object.")
            values = [row.get("title"), row.get("action"), row.get("reason")]
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise LocalModelError("Each suggestion needs a title, action, and reason.")
            suggestions.append(LocalSuggestion(
                title=values[0].strip(),
                action=values[1].strip(),
                reason=values[2].strip(),
            ))
        return suggestions

    def suggest(self, summary: Mapping[str, object]) -> list[LocalSuggestion]:
        """Request short teacher prompts; raw media and identity data are never sent."""
        body = {
            "model": self.model_tag,
            "system": SYSTEM_PROMPT,
            "prompt": json.dumps(dict(summary), sort_keys=True, allow_nan=False),
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.25, "num_predict": 420},
        }
        try:
            if self._http_post is not None:
                response = self._http_post(
                    self.endpoint, json=body, timeout=self.timeout_seconds
                )
            else:
                with requests.Session() as local_session:
                    local_session.trust_env = False
                    response = local_session.post(
                        self.endpoint, json=body, timeout=self.timeout_seconds
                    )
            response.raise_for_status()
            result = response.json()
            generated = result.get("response")
            if not isinstance(generated, str):
                raise LocalModelError("The local model response was not text.")
            parsed = json.loads(generated)
            if not isinstance(parsed, dict):
                raise LocalModelError("The local model response was not a JSON object.")
            return self._validate(parsed)
        except (requests.RequestException, ValueError) as error:
            raise LocalModelError(f"Could not use the local Ollama model: {error}") from error
