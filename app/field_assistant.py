import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import asyncio

load_dotenv()
# Клиент HuggingFace Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

SYSTEM_PROMPT = """
Ты — помощник по написанию интерактивных историй. Твоя задача — по краткой заявке пользователя писать законченные тексты для отдельных полей конфигурации истории (описание истории, герой, NPC, стартовая фраза).
Ты всегда учитываешь текущую конфигурацию истории, которая передана в системных сообщениях в формате JSON.
Всегда строго соблюдай формат и ограничения по длине, указанные в запросе.
Пиши по-русски, литературным, но понятным языком. Не используй списки и не добавляй пояснений от себя.
Не используй много творческих оборотов. Пиши так, будто продолжаешь идею пользователя.
Если пользователь использует какую-то существующую вселенную, уточняй информацию об этой вселенной в интернете.
"""

PLACEHOLDER_INSTRUCTION = (
    " Учитывай общую конфигурацию истории, которая передана в системном сообщении. "
    "Если в тексте нужно упомянуть героя игрока, всегда используй маркер {{user}} вместо имени. "
    "Если нужно упомянуть любого NPC, используй маркер вида {{Имя_NPC}} в двойных фигурных скобках."
)

async def generate_field_value(
    user_prompt: str,
    field_type: str = "description",
    story_config: dict | None = None,
) -> str:
    """
    Генерация текста для конкретного поля (описание истории / герой / npc / стартовая фраза).
    """

    field_map = {
        "story_description": "description",
        "player_description": "player",
        "npc_description": "npc",
        "start_phrase": "start",
    }

    # Нормализуем тип поля до одного из ключей prompt_templates
    internal_field_type = field_map.get(field_type, field_type)

    # Подготовка текста конфигурации истории для системного промта
    config_text = ""
    if story_config is not None:
        try:
            config_json = json.dumps(story_config, ensure_ascii=False, indent=2)
        except TypeError:
            # На случай, если передали не-сериализуемый объект — приводим к строке
            config_json = str(story_config)
        config_text = "\n\nТекущая конфигурация истории (JSON):\n" + config_json

    prompt_templates = {
        "description": (
            "Сделай связное описание истории объёмом не более 15 предложений. "
            "Используй следующую информацию пользователя как черновик, структурируй и уточни её, "
            "но не меняй суть и не добавляй новые факты: {user_input}"
            + PLACEHOLDER_INSTRUCTION
        ),
        "player": (
            "Сделай описание персонажа игрока объёмом не более 5 предложений. "
            "Пиши в третьем лице. Всегда начинай описание текста с имени персонажа.  Используй"
            " следующую информацию пользователя: {user_input}"
            + PLACEHOLDER_INSTRUCTION
        ),
        "npc": (
            "Сделай описание NPC объёмом не более 5 предложений. Пиши в третьем лице. "
            "NPC должен быть индивидуальным, но не добавляй новых фактов. "
            "Информация пользователя: {user_input}"
            + PLACEHOLDER_INSTRUCTION
        ),
        "start": (
            "Сделай стартовую фразу истории объёмом не более 3 предложений. "
            "Это должна быть начальная сцена или реплика, которая плавно вводит игрока в ситуацию. "
            "Используй следующую информацию пользователя: {user_input}"
            + PLACEHOLDER_INSTRUCTION
        ),
    }

    template = prompt_templates.get(internal_field_type, prompt_templates["description"])
    user_message = template.format(user_input=user_prompt.strip())

    system_prompt = SYSTEM_PROMPT + config_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
        messages=messages,
        temperature=0.1
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    async def main():
        # Пример конфигурации истории
        example_story_config = {
        }

        user_prompt = ""

        print("\n=== Тест: персонаж игрока ===")
        player = await generate_field_value(
            user_prompt="Рок Мунстоун - персонаж из вселенной Genshin Impact, близкий друг {{durin}}. Помогает ему осваиваться в Тейвате",
            field_type="player_description",
            story_config=example_story_config,
        )
        print(player)

    asyncio.run(main())