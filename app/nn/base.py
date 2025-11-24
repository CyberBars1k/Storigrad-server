from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from app.schemas import InferenceRequest

class Module(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def score(self, req: InferenceRequest) -> float:
        """Return confidence in [0,1] that this module should handle the request."""

    @abstractmethod
    def run(self, req: InferenceRequest) -> str:
        """Produce module's contribution to the reply."""

    def decide(self, req: InferenceRequest, thr: float = 0.5) -> Tuple[bool, float]:
        s = self.score(req)
        return (s >= thr, s)