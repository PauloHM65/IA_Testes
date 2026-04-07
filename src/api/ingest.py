"""Router: ingestao de documentos + estado de ingestao."""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException

from src.api.schemas import IngestResponse
from src.config import ServiceConfig
from src.ingest import RedisIngester
from src.logging.file_logger import get_logger
from src.pipeline import invalidate_pipeline

router = APIRouter(prefix="/api/services", tags=["ingest"])

# Estado de ingestao por servico
_ingest_status: dict[str, str] = {}


def init_ingest_status(service_names: list[str]):
    """Inicializa status de ingestao para os servicos."""
    for name in service_names:
        _ingest_status[name] = "idle"


def trigger_ingest(service: str, config: ServiceConfig):
    """Dispara ingestao em background thread."""
    if _ingest_status.get(service) == "running":
        return

    def _run():
        _ingest_status[service] = "running"
        try:
            RedisIngester(config, get_logger()).run(force=True)
            invalidate_pipeline(service)
            _ingest_status[service] = "done"
        except Exception as e:
            _ingest_status[service] = f"error:{e}"

    threading.Thread(target=_run, daemon=True).start()


@router.get("/{service}/ingest/status")
def ingest_status(service: str):
    """Retorna status da ingestao do servico."""
    return {"status": _ingest_status.get(service, "idle")}


@router.post("/{service}/ingest")
def ingest(service: str, force: bool = False):
    """Executa ingestao de documentos."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    try:
        result = RedisIngester(config, get_logger()).run(force=force)
        if result is None:
            return IngestResponse(status="skipped", message="Documentos nao mudaram.")
        return IngestResponse(status="ok", message="Ingestao concluida.")
    except Exception as e:
        raise HTTPException(500, str(e))
