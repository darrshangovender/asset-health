"""Optional LLM narrative generator (``pip install "asset-health[llm]"``).

Nothing in the core pipeline, the tests or the benchmark imports this module, and no API key is
needed anywhere else. The generator only ever sees the structured work order, and whatever it
returns still has to pass ``check_narrative`` before it is attached (see ``narrate``).

The model name is a required argument on purpose: model identifiers are retired over time, so a
default baked into a library goes stale.
"""

from __future__ import annotations

import json
from typing import Any

from asset_health.models import WorkOrder

SYSTEM_PROMPT = (
    "You write short maintenance notes for a reliability engineer. Use ONLY facts present in the "
    "JSON work order you are given: do not add numbers, dates, equipment names, causes or "
    "recommendations that are not in it. Mention the asset, the most likely fault, the key "
    "evidence with its values, the remaining-life interval, the latest safe date and the "
    "recommended action. At most 110 words, plain text."
)


class AnthropicNarrator:
    """Narrative generator backed by the Anthropic Messages API."""

    def __init__(self, model: str, client: Any | None = None, max_tokens: int = 400) -> None:
        if not model:
            raise ValueError("pass the model identifier explicitly")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError('install the extra: pip install "asset-health[llm]"') from exc
            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, work_order: WorkOrder) -> str:
        payload = json.loads(work_order.model_dump_json(exclude={"narrative", "narrative_source"}))
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content).strip()
