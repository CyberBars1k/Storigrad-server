import os
from typing import Any
from dotenv import load_dotenv
import copy
import json
import openai
from sqlalchemy.orm import Session

from . import models, story as story_crud

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("YANDEX_CLOUD_API_KEY"),
    base_url=os.getenv("YANDEX_CLOUD_BASE_URL", "https://rest-assistant.api.cloud.yandex.net/v1"),
    project=os.getenv("YANDEX_CLOUD_PROJECT"),
)


def _set_story_config_value(db: Session, story: models.Story, key: str, value: Any) -> None:
    """Safely updates JSON-like config and commits the change."""
    cfg = copy.deepcopy(story.config or {})
    cfg[key] = value
    story.config = cfg
    db.add(story)
    db.commit()


def generate_story_step(
    db: Session,
    story_id: int,
    user_id: int,
    user_input: str,
    mode: str = "dialogue",  # режим: dialogue / narration / directive
) -> str:
    """
    Основной вход для Storyteller-mini в серверном режиме.

    Использует Yandex Cloud Responses API с agent prompt.
    Поддерживает передачу previous_response_id для контекста и непрерывности диалога.
    """

    # 1. Находим историю и проверяем владельца
    story = (
        db.query(models.Story)
        .filter(
            models.Story.id == story_id,
            models.Story.owner_id == user_id,
        )
        .first()
    )
    if not story:
        raise ValueError("История не найдена или нет доступа")

    config = story.config or {}
    story_description = config.get("story_description", "")
    player_description = config.get("player_description", {})
    npc_description = config.get("NPC_description", [])
    yc_agent_prompt_id = config.get("yc_agent_prompt_id") or os.getenv("YANDEX_CLOUD_AGENT_PROMPT_ID")
    yc_previous_response_id = config.get("yc_previous_response_id")

    if not yc_agent_prompt_id:
        raise ValueError("Не задан yc_agent_prompt_id / YANDEX_CLOUD_AGENT_PROMPT_ID")

    if not os.getenv("YANDEX_CLOUD_API_KEY"):
        raise ValueError("Не задан YANDEX_CLOUD_API_KEY")

    if not os.getenv("YANDEX_CLOUD_PROJECT"):
        raise ValueError("Не задан YANDEX_CLOUD_PROJECT")

    # 3. Resolve user name from player_description (agent variables expect plain strings)
    resolved_user_name = ""
    user_field_val: Any = ""
    if isinstance(player_description, dict):
        user_field_val = player_description.get("user", "")
    elif isinstance(player_description, str):
        user_field_val = player_description

    if isinstance(user_field_val, str):
        if "—" in user_field_val:
            resolved_user_name = user_field_val.split("—", 1)[0].strip()
        elif "-" in user_field_val:
            resolved_user_name = user_field_val.split("-", 1)[0].strip()
        else:
            resolved_user_name = user_field_val.strip()

    # 4. Build variables dict for agent prompt
    variables = {
        "NPC_description": json.dumps(npc_description, ensure_ascii=False),
        "story_description": str(story_description),
        "user": resolved_user_name,
        "player_description": json.dumps(player_description, ensure_ascii=False),
        "mode": str(mode),
    }

    # 5. Build input string
    input_text = f"MODE: {mode}\nUSER_TURN: {user_input}".strip()

    # 6. Call Yandex Cloud responses.create
    try:
        kwargs = {
            "prompt": {"id": yc_agent_prompt_id, "variables": variables},
            "input": input_text,
        }
        if yc_previous_response_id:
            kwargs["previous_response_id"] = yc_previous_response_id

        response = client.responses.create(**kwargs)
        # Yandex docs show both `response.output_text` and `response.output[0].content[0].text` depending on mode.
        story_text = getattr(response, "output_text", None)
        if not story_text:
            try:
                story_text = response.output[0].content[0].text
            except Exception:
                story_text = ""
    except Exception as e:
        print("Yandex Cloud error in generate_story_step:", repr(e))
        return "Сейчас рассказчик недоступен, попробуйте ещё раз позже."

    # 8. Save the turn in DB
    story_crud.add_turn(
        db=db,
        story_id=story_id,
        user_text=user_input,
        model_text=story_text,
    )

    # 9. Persist new response id for continuity
    new_response_id = getattr(response, "id", None)
    if new_response_id:
        _set_story_config_value(db, story, "yc_previous_response_id", new_response_id)

    return story_text
