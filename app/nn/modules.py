import re
from typing import List
from app.schemas import InferenceRequest
from .base import Module

class GreetingModule(Module):
    def __init__(self):
        super().__init__("greeting")

    def score(self, req: InferenceRequest) -> float:
        txt = req.message.lower()
        return 1.0 if any(w in txt for w in ["привет", "здрав", "hello", "hi"]) else 0.0

    def run(self, req: InferenceRequest) -> str:
        return "Принято. Вы в Сториграде. Кратко опишите, что хотите сделать в мире истории."

class LoreModule(Module):
    def __init__(self):
        super().__init__("lore")

    def score(self, req: InferenceRequest) -> float:
        return 1.0 if re.search(r"(мир|локаци|персонаж|сеттинг)", req.message.lower()) else 0.0

    def run(self, req: InferenceRequest) -> str:
        return "Опишите локацию, время, цель. Я предложу три стартовых сцены."

class ActionModule(Module):
    def __init__(self):
        super().__init__("action")

    def score(self, req: InferenceRequest) -> float:
        txt = req.message.lower()
        return 0.8 if any(w in txt for w in ["идти", "атак", "взять", "осмотреть", "open", "go"]) else 0.0

    def run(self, req: InferenceRequest) -> str:
        return "Действие принято. Сформирую ответ ведущего и последствия." 

class FallbackModule(Module):
    def __init__(self):
        super().__init__("fallback")

    def score(self, req: InferenceRequest) -> float:
        return 1.0  # always available

    def run(self, req: InferenceRequest) -> str:
        return "Опишите цель. Доступны: создание мира, выбор роли, действие."


def default_modules() -> List[Module]:
    return [GreetingModule(), LoreModule(), ActionModule(), FallbackModule()]