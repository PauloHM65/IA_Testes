"""Schemas Pydantic compartilhados entre routers."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    service: str = "default"
    question: str


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[str] = []
    raw_count: int = 0
    rerank_count: int = 0


class IngestResponse(BaseModel):
    status: str
    message: str


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    tipo: str  # "materia" ou "exercicio"


class ModelSwitchRequest(BaseModel):
    provider: str
    model: str
