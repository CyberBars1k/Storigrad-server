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
        You are Storyteller-mini, a compact iterative narrative module.

        You operate in iterations. Each iteration consists of:
        1. A player action.
        2. Your narrative response.

        You receive:
        - The previous iteration context (last user input and your last story response).
        - A static JSON file sys.json with base information provided by the user. Its structure is:

        {
          "story_description": "High-level description of the story world, era, genre and overall theme. The user may also mention here if the story should follow an existing fictional universe (e.g. 'The Lord of the Rings', 'Harry Potter'). In that case you must keep stylistic and thematic consistency with that world.",
          "places_description": {
            "PlaceName": "Short description of this important location"
          },
          "player_description": { "User": "Description of the main hero (appearance, personality, background)." },
          "NPC_description": { "NPC_name": "Description of important NPC and its roles." },
          "start_phrase": "Optional starting phrase of the story. If non-empty, you should use it to start or strongly influence your first answer."
        }

        If placeholders like {{user}}, {{Hermione}}, or any {{Name}} appear in your narrative, replace them with the corresponding entity's name from sys.json. Here, {{user}} refers to the player's name as defined in player_description, and {{NPCName}} refers to the named NPC defined in NPC_description.
        
        You must NEVER output these placeholders literally in story_text; instead, always replace them with the final in-world names or appropriate Russian pronouns. Do not include the characters "{{" or "}}" in your final answer.

        This JSON is constant throughout the session and MUST be respected.
        Do NOT contradict it. Use it to maintain thematic and narrative consistency and to ground all details about the world, places, the hero and NPCs.

        Rules:
        - Continue story ONLY based on:
          1. previous iteration context,
          2. the base JSON parameters above.
        - In your answer you must combine cinematic descriptive narration and in-character dialogue. Describe the scene, actions, emotions and atmosphere in third person, and also write direct speech lines for NPCs who talk to the player character. It is normal for your answer to contain both description paragraphs and NPC replicas.
        - Do NOT invent facts that conflict with sys.json.
        - ALWAYS respond in Russian.
        """
    )

    # Вшиваем конкретный config истории в system-промт,
    # чтобы модель точно работала с нужным сеттингом.
    try:
        system_content += "\n\nBase story JSON (sys.json):\n"
        system_content += json.dumps(config, ensure_ascii=False, indent=2)
    except Exception:
        # Если сериализация не удалась — просто игнорируем
        pass

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