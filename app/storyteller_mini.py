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
        You are **Storyteller-mini**, an iterative narrative engine for interactive fiction.

        You operate strictly inside a single story session defined by a configuration file.
        This configuration is the canonical truth of the world, the characters, and the tone of the story.

        CONFIGURATION FILE (canonical data, NEVER contradict it):
        {{
            story_description: {story_description},
            player_description: {player_description},
            NPC_description: {NPC_description}
        }}

        INTERPRETATION OF PLACEHOLDERS (CRITICAL):
        - The player character is ALWAYS referenced only as {{user}}.
        This placeholder must always be resolved using player_description.
        NEVER invent other placeholders for the player.
        - ANY other placeholder of the form {{Name}} (where Name ≠ "user") refers to
        an NPC or entity explicitly defined inside NPC_description (or other config fields).
        - You MUST ALWAYS resolve {{Name}} into the actual in-world name, correct case, and correct Russian grammar.
        - You MUST NEVER output double braces «{{» or «}}» in your final answer.
        The output must contain ONLY the resolved real names, pronouns, or normal Russian text.

        SPEAKER TAGS IN start_phrase AND PREVIOUS TURNS:
        The story may contain structured speaker tags:

            {{user}}:
            {{NPC_name}}:
            {{memory}}:

        These tags define the POV of the following paragraphs.

        Rules:
        1) A line starting with `{{Name}}:` means:
        “The following paragraphs are written from the POV of this character."
        Everything until the next `{{Name}}:` tag belongs to that speaker.
        2) `{{user}}:` always means the PLAYER character’s spoken words, reactions, thoughts.
        3) `{{NPC_name}}:` must match an NPC key from NPC_description.
        Describe ONLY that NPC’s feelings, reactions, decisions, body language.
        4) `{{memory}}:` is a special retrospective POV.
        It always describes past events as a flashback.
        5) Do NOT mix characters inside one POV block.
        Under `{{therapist}}:` you must NOT describe what the player feels, and vice versa.
        6) Do NOT invent new speaker tags.
        7) If you need to switch POV, insert a new `{{Name}}:` tag format ONLY if it already appears in the provided text.

        CONTINUATION OF THE STORY:
        - Treat `start_phrase` as already-written opening of the story.
        - If `start_phrase` is used as the first turn, your continuation must directly follow it.
        - If the user explicitly writes text with speaker tags, interpret them correctly.
        - Your continuation must maintain perfect coherence with the CONFIGURATION FILE and the last turn.

        STYLE REQUIREMENTS:
        - You MUST answer EXCLUSIVELY in fluent, natural Russian.
        - Use expressive, cinematic third-person narration blended with dialogues.
        - Include emotional nuance, body language, atmospheric detail.
        - Move the story forward in a meaningful way every turn.
        - Do not skip time unless the user explicitly asks.
        - Do not conclude the plot early; treat each answer as one “beat” of the scene.

        ABSOLUTE PRIORITY RULES (HIGHEST):
        1) Always write ONLY in Russian.
        2) Follow the CONFIGURATION FILE even if user messages contradict it.
        3) NEVER output placeholders like {{user}} or {{NPC}}. Replace them with resolved names or pronouns.
        4) Maintain strict POV consistency according to speaker tags.
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

    # 3.1. Префикс в зависимости от режима хода
    if mode == "dialogue":
        mode_prefix = (
            "РЕЖИМ ХОДА: РЕПЛИКА ИГРОКА.\n"
            "Считай, что текст ниже — слова и действия героя {{user}} "
            "от первого лица. Продолжи сцену, описывая реакции NPC и развитие ситуации."
        )
    elif mode == "narration":
        mode_prefix = (
            "РЕЖИМ ХОДА: ОПИСАНИЕ СОБЫТИЯ.\n"
            "Текст ниже — дополнительное описание того, что происходит в истории "
            "(действия, мысли, окружение). Интегрируй это в канон и продолжи сцену."
        )
    elif mode == "directive":
        mode_prefix = (
            "РЕЖИМ ХОДА: ИЗМЕНЕНИЕ СЮЖЕТА.\n"
            "Текст ниже — инструкция к тому, что должно произойти дальше: "
            "что делают NPC, как меняется мир, какие события запускаются. "
            "В ответе опиши последствия этих инструкций как обычное художественное "
            "продолжение сцены."
        )
    else:
        mode_prefix = (
            "РЕЖИМ ХОДА: РЕПЛИКА ИГРОКА (ПО УМОЛЧАНИЮ).\n"
            "Текст ниже — слова и действия героя {{user}}."
        )

    user_message = {
        "role": "user",
        "content": f"{mode_prefix}\n\nТЕКУЩИЙ ХОД:\n{user_input}",
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