import os
import json
import asyncio
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import openai

load_dotenv()

# Yandex Cloud AI Studio (Assistant Responses API)
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_CLOUD_PROJECT_ID = os.getenv("YANDEX_CLOUD_PROJECT")
# Optional: prompt/assistant id if you configured it in AI Studio
YANDEX_FIELD_ASSISTANT_PROMPT_ID = os.getenv("YANDEX_FIELD_ASSISTANT_PROMPT_ID")

if not YANDEX_CLOUD_API_KEY:
    raise RuntimeError("YANDEX_CLOUD_API_KEY is not set")

client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    project=YANDEX_CLOUD_PROJECT_ID,
)



def _extract_text_from_response(resp: Any) -> str:
    """Best-effort extraction of text from Responses API."""
    # Newer OpenAI SDKs often expose `output_text` convenience
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # Fallback: try to traverse response.output
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        chunks.append(t)
        joined = "\n".join(chunks).strip()
        if joined:
            return joined

    # Last resort
    return str(resp).strip()


def _json_loads_strict(text: str) -> Dict[str, Any]:
    """Parse JSON from model output. Accepts raw JSON or fenced code blocks."""
    s = (text or "").strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        s = s.strip("`")
        # If it was ```json ...
        s = s.replace("json\n", "", 1)

    # Try direct parse
    return json.loads(s)


def _validate_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Light validation + normalization to the expected keys."""
    if not isinstance(obj, dict):
        raise ValueError("Assistant output is not a JSON object")

    # Required keys
    for k in ("story_description", "player_description", "npc_description"):
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["story_description"], str):
        raise ValueError("story_description must be a string")

    pd = obj["player_description"]
    if not isinstance(pd, dict) or "user" not in pd or not isinstance(pd.get("user"), str):
        raise ValueError("player_description must be an object with string field 'user'")

    nd = obj["npc_description"]
    if not isinstance(nd, dict) or not nd:
        raise ValueError("NPC_description must be a non-empty object")
    for name, desc in nd.items():
        if not isinstance(name, str) or not isinstance(desc, str):
            raise ValueError("NPC_description must be {name: string_description}")

    return obj


async def generate_story_config(
    genre: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Generate full story config from (genre, user_prompt). Returns a parsed JSON dict."""

    genre = (genre or "").strip()
    user_prompt = (user_prompt or "").strip()

    if not genre:
        raise ValueError("genre is required")
    if not user_prompt:
        raise ValueError("user_prompt is required")

    # The model should receive only genre + user prompt.
    # Keep the user message simple and explicit.
    user_message = (
        f"Жанр: {genre}\n"
        f"Запрос пользователя: {user_prompt}\n\n"
    )

    # Preferred path: use prompt id if provided (AI Studio Agent/Prompt)
    if YANDEX_FIELD_ASSISTANT_PROMPT_ID:
        resp = client.responses.create(
            prompt={
                "id": YANDEX_FIELD_ASSISTANT_PROMPT_ID,
            },
            input=user_message,
        )

    text = _extract_text_from_response(resp)

    try:
        obj = _json_loads_strict(text)
        return _validate_schema(obj)
    except Exception:
        # Repair pass: force the assistant to output valid JSON only
        repair_user = (
            "Твой предыдущий ответ не является валидным JSON по схеме. "
            "Верни ТОЛЬКО валидный JSON-объект строго по формату (без markdown, без пояснений).\n\n"
            f"Жанр: {genre}\n"
            f"Запрос пользователя: {user_prompt}"
        )

        if YANDEX_FIELD_ASSISTANT_PROMPT_ID:
            resp2 = client.responses.create(
                prompt={
                    "id": YANDEX_FIELD_ASSISTANT_PROMPT_ID,
                },
                input=repair_user,
            )

        text2 = _extract_text_from_response(resp2)
        obj2 = _json_loads_strict(text2)
        return _validate_schema(obj2)


# Backwards-compatible wrapper name (if other code imports it)
async def generate_field_values(genre: str, user_prompt: str) -> Dict[str, Any]:
    return await generate_story_config(genre=genre, user_prompt=user_prompt)