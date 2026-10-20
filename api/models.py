from pydantic import BaseModel
from typing import Any



class GenerateRequest(BaseModel):
    question: str
    graph: str 
    schema_mode: str
    llm: str | None = None
    execute: bool = True

class GenerateResponse(BaseModel):
    question: str
    graph: str
    schema_mode: str
    generated_sparql: str
    execution_time: float
    query_result: dict[str, Any] | None = None
    query_execution_time: float | None = None