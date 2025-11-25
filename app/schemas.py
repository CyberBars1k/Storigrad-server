from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class InferenceRequest(BaseModel):
    message: str = Field(..., description="User input text")
    context: Optional[List[str]] = Field(default=None, description="Optional short context lines")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Client metadata")

class TraceItem(BaseModel):
    module: str
    accepted: bool
    output: str
    score: float

class InferenceResponse(BaseModel):
    reply: str
    modules: List[str]
    trace: List[TraceItem]
    latency_ms: int

class HealthResponse(BaseModel):
    status: str

class StoryStepIn(BaseModel):
    story_id: int = Field(..., description="ID of the story to continue")
    user_input: str = Field(..., description="Text input from user")