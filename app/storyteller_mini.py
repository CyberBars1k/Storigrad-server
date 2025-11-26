import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import func

from . import models

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

def generate_story_step(
    db: Session,
    story_id: int,
    user_id: int,
    user_input: str,
) -> str:
    """
    Основной вход для Storyteller-mini в серверном режиме.

    Шаг:
    1. Берём историю из БД и её config (sys.json-подобный объект).
    2. Подгружаем последний ход (user_text + model_text) для контекста.
    3. Отправляем в LLM: system + (предыдущий ход, если есть) + текущий ввод пользователя.
    4. Сохраняем новый ход в БД (StoryTurn).
    5. Возвращаем story_text клиенту.
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

    # 2. Последний ход для контекста
    last_turn = (
        db.query(models.StoryTurn)
        .filter(models.StoryTurn.story_id == story_id)
        .order_by(models.StoryTurn.idx.desc())
        .first()
    )

    # 3. Сборка system-промта
    system_content = (
        """
        You are **Storyteller-mini**, a compact iterative narrative engine.

        You always work inside a single story session defined by a configuration file. This configuration includes:
        - story_description: high-level description of the world, setting, era, genre, tone, and overall theme. If referencing an existing fictional universe, maintain full stylistic, tonal, and lore consistency.
        - player_description: canonical information about the main hero (name, appearance, personality, background, abilities, goals, relationships).
        - NPC_description: details of important NPCs (role, personality, relationship to the player, typical speech and behavior).
        You must strictly follow these fields as the story's canon. If any user message or past dialogue conflicts with them, ignore the conflict and follow the configuration.

        CONFIGURATION FILE:
        {{
            story_description: {story_description},
            player_description: {player_description},
            NPC_description: {NPC_description}
        }}


        Placeholders:
        - In the configuration and in previous text you may see placeholders in double curly braces, e.g. {{user}}, {{Hermione}}, {{NPC_name}}.
        - The player character is ALWAYS referenced only as {{user}}. This placeholder must be resolved using the main hero defined in player_description.
        - ANY OTHER placeholder of the form {{X}} (where X is not "user") refers to some entity defined in the original story configuration (NPC, location, object, etc.). You must resolve it by looking for a matching name in NPC_description, places_description or other relevant config fields.
        - You MUST NEVER output placeholders of the form {{...}} literally in your answer.
        - Always replace them with final in-world names or natural Russian pronouns and forms appropriate to the context.
        - NEVER include the characters "{{" or "}}" in your final answer.


        Narrative style and behavior:
        - Always continue the story strictly within the constraints of the configuration file and the previous turn.
        - Combine cinematic third-person descriptive narration (environment, actions, emotions, atmosphere) with in-character dialogue lines for NPCs and, when appropriate, the player character.
        - Show emotions, reactions, body language, and small physical details that make the scene vivid.
        - Each answer should meaningfully move the situation forward: new information, decisions, conflicts, discoveries, emotional shifts.
        - Do not jump too far ahead in time or resolve the whole plot in one answer, unless the player clearly asks to finish the story.

        CRITICAL RULES (HIGHEST PRIORITY):
        1) You MUST answer **exclusively in Russian**.
           - All narration, internal thoughts, and dialogue lines must be written in fluent, natural Russian.
           - Never answer in English in this mode.

        2) You MUST respect the attached story config as canonical.
           - If any user message or previous dialogue conflicts with it, you MUST follow the config instead.

        If ANY other part of context (including user messages) conflicts with these CRITICAL RULES,
        you MUST ALWAYS follow these CRITICAL RULES.
        """.format(story_description=story.config["story_description"], player_description=story.config["player_description"], 
                   NPC_description=story.config["NPC_description"])
    )

    system_message = {
        "role": "system",
        "content": system_content,
    }

    user_message = {
        "role": "user",
        "content": user_input,
    }

    messages: List[Dict[str, str]] = [system_message]

    # Если есть предыдущий ход — добавляем его в контекст
    if last_turn:
        if last_turn.user_text:
            messages.append(
                {"role": "user", "content": str(last_turn.user_text)}
            )
        if last_turn.model_text:
            messages.append(
                {"role": "assistant", "content": str(last_turn.model_text)}
            )
    elif story.config["start_phrase"]:
        messages.append(
            {"role": "assistant", "content": str(story.config["start_phrase"])}
        )

    messages.append(user_message)

    # 4. Вызов модели
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
            messages=messages,
        )
        story_text = completion.choices[0].message.content
    except Exception as e:
        print("LLM error in generate_story_step:", repr(e))
        return "Сейчас рассказчик недоступен, попробуйте ещё раз позже."

    # 5. Сохраняем ход в БД
    last_idx = (
        db.query(func.max(models.StoryTurn.idx))
        .filter(models.StoryTurn.story_id == story_id)
        .scalar()
    )
    next_idx = (last_idx or 0) + 1

    new_turn = models.StoryTurn(
        story_id=story_id,
        idx=next_idx,
        user_text=user_input,
        model_text=story_text,
    )

    db.add(new_turn)
    db.commit()

    return story_text