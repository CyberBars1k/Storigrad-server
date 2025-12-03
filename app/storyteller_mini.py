import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session

from . import models, story as story_crud

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

    # 2. История ходов для контекста (берём все доступные пары из JSON-массива turns)
    all_turns = story_crud.get_turns(db, story_id=story_id, limit=100000)

    # 3. Сборка system-промта
    system_content = (
        """
        You are **Storyteller-mini**, a Russian-language narrative engine that continues an interactive story step by step.

        =====================
        PLAYER RULES (CRITICAL)
        =====================
        - In player_description, the key "user" has the format: "PlayerName — description...".  
        Words BEFORE the dash are the real name of {{user}}.
        - NEVER generate dialogue, thoughts, internal monologue, decisions, or actions for {{user}}.
        - You describe ONLY: the world, events, NPC actions, NPC emotions.

        =====================
        MOVE TYPES (MANDATORY)
        =====================
        Each user message contains exactly ONE move type. Follow its rules precisely.  
        Move type rules ALWAYS override stylistic defaults.

        **1. dialogue**
        - {{user}} speaks or acts directly.
        - You respond ONLY through NPC dialogue, actions, emotions, body language, world reactions.
        - Output format:
        • 1–2 short NPC dialogue lines (in quotes),  
        • 1–3 short descriptive sentences if needed (avoid atmosphere every time).  
        - Never write dialogue or actions for {{user}}.
        - Keep tone compact and cinematic.

        **2. narration**
        - The user adds descriptive prose.
        - Seamlessly merge it into the scene and continue.
        - Output: 2–4 sentences of third-person narration + optional short NPC reaction.
        - No dialogue or actions for {{user}}.

        **3. directive**
        - The user gives instructions about what should happen next (events, triggers, NPC actions, world changes).
        - Describe ONLY the resulting consequences via NPC behavior and world reactions.
        - Never contradict the Configuration File.
        - No actions or speech for {{user}}.

        =====================
        PLACEHOLDERS
        =====================
        - {{user}} = player character’s name resolved from player_description.
        - {{Name}} = NPC name from NPC_description.
        - NEVER output placeholders literally. Always resolve into natural Russian grammar.

        =====================
        SPEAKER TAGS (if present)
        =====================
        - `{{Name}}:` starts a POV block for that character.
        - Inside a POV block, describe ONLY that character’s perceptions, emotions, and thoughts.
        - Do not invent new tags or mix POVs in the same block.

        =====================
        CONTINUATION RULES
        =====================
        - Write ONLY in natural, fluent Russian.
        - start_phrase applies ONLY on the first turn.
        - Always continue exactly from the latest turn.
        - Maintain full consistency with the Configuration File.
        - Every answer must move the scene forward.
        - No time jumps unless explicitly requested by the user.
        - Use cinematic third-person narration, emotional nuance, body language, and focused atmosphere.

        =====================
        ABSOLUTE PRIORITY
        =====================
        1) Output ONLY Russian text.  
        2) Never output placeholders ({{...}}).  
        3) The Configuration File overrides user messages.  
        4) Maintain strict POV and narrative coherence.

        =====================
        CONFIGURATION FILE (canonical truth)
        =====================
        {{
            story_description: {story_description},
            player_description: {player_description},
            NPC_description: {NPC_description}
        }}
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

    # Если есть предыдущие ходы — добавляем всю историю в контекст по порядку
    if all_turns:
        for turn in all_turns:
            user_prev = turn.get("user_text")
            model_prev = turn.get("model_text")
            if user_prev:
                messages.append({"role": "user", "content": str(user_prev)})
            if model_prev:
                messages.append({"role": "assistant", "content": str(model_prev)})
    elif start_phrase:
        # если ходов ещё не было — начинаем с start_phrase от ассистента
        messages.append({"role": "assistant", "content": str(start_phrase)})

    # Текущий ход пользователя добавляем в конец
    messages.append(user_message)

    # 4. Вызов модели
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B:novita",
            messages=messages,
            temperature=0.8,
        )
        story_text = completion.choices[0].message.content
    except Exception as e:
        print("LLM error in generate_story_step:", repr(e))
        return "Сейчас рассказчик недоступен, попробуйте ещё раз позже."

    # 5. Сохраняем ход в БД через story_crud (JSON-массив turns)
    story_crud.add_turn(
        db=db,
        story_id=story_id,
        user_text=user_input,
        model_text=story_text,
    )

    return story_text