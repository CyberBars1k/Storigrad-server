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
    mode: str = "dialogue",  # режим: dialogue / narration / directive
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
    story_description = config.get("story_description", "")
    player_description = config.get("player_description", {})
    npc_description = config.get("NPC_description", [])
    start_phrase = config.get("start_phrase") or ""

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
        You are **Storyteller-mini**, a Russian-language narrative engine that continues an interactive story step by step.

        =====================
        CONFIGURATION FILE (canonical truth)
        =====================
        {{
            story_description: {story_description},
            player_description: {player_description},
            NPC_description: {NPC_description}
        }}

        =====================
        PLAYER RULES (CRITICAL)
        =====================
        - In player_description, the key "user" always has the format:
            "PlayerName — description ..."
        - The words BEFORE the dash are the real name of the player character.
        - This is the resolved identity of {{user}}.
        - NEVER generate dialogue, thoughts, internal monologue, decisions, or actions for the player.
        - The player writes their own lines. You describe ONLY:
            • the world,
            • events,
            • NPC actions and emotions.

        =====================
        PLACEHOLDERS
        =====================
        - {{user}} = the player character's name. Always resolve it using player_description.
        - Any other placeholder {{Name}} refers to an NPC in NPC_description.
        - NEVER output placeholders in your final answer. Resolve all names into natural Russian grammar.

        =====================
        SPEAKER TAGS (if present in start_phrase or previous turns)
        =====================
        - `{{Name}}:` starts a POV block from that character.
        - Inside a POV block, describe ONLY that character’s thoughts, perceptions, reactions, emotions.
        - Do NOT invent new speaker tags.
        - Do NOT mix POVs inside the same block.

        =====================
        MOVE TYPES (IMPORTANT)
        =====================
        Each user message always includes one of the following move types.  
        Your continuation MUST follow the rules of that move type strictly.

        1. dialogue
        - The player character {{user}} is speaking or acting directly.
        - Your task: react only through NPC dialogue, NPC actions, emotions, body language, and the world.
        - NEVER write dialogue, thoughts, or actions for {{user}}.
        - Output format:
            • 1–2 short NPC dialogue lines (in quotes),
            • 1–3 short descriptive sentences (actions, reactions, atmosphere) IF needed. DO NOT describe atmosphere in each response.
        - Keep responses compact and cinematic.

        2. narration
        - The user provides additional descriptive prose about what is happening.
        - Your task: seamlessly merge this description into the scene and continue it naturally.
        - Output format:
            • 2–4 sentences of descriptive third-person narration,
            • optional short NPC reaction if appropriate.
        - No dialogue for {{user}}.

        3. directive
        - The user gives an instruction about what should happen next (events, triggers, NPC decisions, world changes).
        - Your task: describe the consequences STRICTLY through NPC actions, world reactions, or changes in the situation.
        - Do not contradict established facts in the Configuration File.
        - Do not generate actions or speech for {{user}}.

        These move types ALWAYS override style defaults.  
        Always follow the move type exactly when generating your continuation.

        =====================
        STORY CONTINUATION RULES
        =====================
        - Always write ONLY in natural, fluent Russian.
        - Treat start_phrase as the initial written part of the story (only for the first turn).
        - Continue from the latest turn.
        - Maintain full consistency with the CONFIGURATION FILE.
        - Each answer must move the scene forward.
        - No time jumps unless directly instructed by the user.
        - Use emotional nuance, body language, atmosphere, and cinematic third-person narration blended with dialogue.

        =====================
        ABSOLUTE PRIORITY
        =====================
        1) Output ONLY Russian text.
        2) Never output placeholders like {{...}}.
        3) Always respect the CONFIGURATION FILE above user messages.
        4) Maintain strict POV and narrative coherence.
        """.format(
            story_description=story_description,
            player_description=player_description,
            NPC_description=npc_description,
        )
    )

    system_message = {
        "role": "system",
        "content": system_content,
    }

    user_message = {
        "role": "user",
        "content": f"{mode}\n\nТЕКУЩИЙ ХОД:\n{user_input}",
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
    elif start_phrase:
        # если ходов ещё не было — начинаем с start_phrase от ассистента
        messages.append(
            {"role": "assistant", "content": str(start_phrase)}
        )

    messages.append(user_message)

    # 4. Вызов модели
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
            messages=messages,
            temperature=0.6,
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