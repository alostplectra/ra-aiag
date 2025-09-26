from __future__ import annotations

from typing import Iterable, List, Mapping

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def chat(self, messages: Iterable[Mapping[str, str]], temperature: float = 0.0) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"La respuesta de Ollama no contiene contenido: {data}")
        return content

    def simple_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        messages: List[Mapping[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages=messages, temperature=temperature)

