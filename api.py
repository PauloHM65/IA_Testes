"""API REST para o DRAG — expoe o pipeline via HTTP/WebSocket."""

from __future__ import annotations

import asyncio
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from src.config import ServiceConfig, list_services, get_default_service
from src.pipeline import DragPipeline, get_pipeline, _pipeline_cache
from src.ingest import run_ingest


# ---------------------------------------------------------------------------
# Estado de ingestao (por servico)
# ---------------------------------------------------------------------------

_ingest_status: dict[str, str] = {}  # "idle" | "running" | "done" | "error:msg"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    services = list_services()
    for name in services:
        _ingest_status[name] = "idle"
    if services:
        default = get_default_service()
        try:
            config = ServiceConfig.load(default)
            get_pipeline(config)
        except Exception:
            pass
    yield


app = FastAPI(
    title="DRAG API",
    description="Retrieval-Augmented Generation com Redis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints: Servicos
# ---------------------------------------------------------------------------

@app.get("/api/services")
def get_services():
    """Lista servicos disponiveis."""
    services = list_services()
    result = []
    for name in services:
        try:
            config = ServiceConfig.load(name)
            result.append({
                "name": config.name,
                "display_name": config.display_name,
                "pipeline_steps": list(config.pipeline_steps),
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model,
            })
        except Exception:
            result.append({"name": name, "display_name": name, "pipeline_steps": []})
    return result


# ---------------------------------------------------------------------------
# Endpoints: Arquivos
# ---------------------------------------------------------------------------

@app.get("/api/services/{service}/files")
def list_files(service: str):
    """Lista arquivos de materias e exercicios do servico."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    files = []
    for tipo, dir_path in [("materia", config.materias_dir), ("exercicio", config.exercicios_dir)]:
        p = Path(dir_path)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".pdf", ".txt"):
                files.append(FileInfo(
                    name=f.name,
                    path=str(f),
                    size=f.stat().st_size,
                    tipo=tipo,
                ))

    return files


@app.post("/api/services/{service}/upload")
async def upload_file(service: str, tipo: str = "materia", file: UploadFile = File(...)):
    """Upload de arquivo para materias ou exercicios do servico."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    if tipo == "exercicio":
        dest_dir = Path(config.exercicios_dir)
    else:
        dest_dir = Path(config.materias_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingestao em background
    _trigger_ingest(service, config)

    return {"status": "ok", "file": file.filename, "path": str(dest)}


@app.delete("/api/services/{service}/files/{filename}")
def delete_file(service: str, filename: str, tipo: str = "materia"):
    """Remove arquivo do servico."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    if tipo == "exercicio":
        dest_dir = Path(config.exercicios_dir)
    else:
        dest_dir = Path(config.materias_dir)

    target = dest_dir / filename
    if not target.exists():
        raise HTTPException(404, f"Arquivo '{filename}' nao encontrado.")

    target.unlink()
    _trigger_ingest(service, config)

    return {"status": "ok", "message": f"'{filename}' removido."}


# ---------------------------------------------------------------------------
# Ingestao em background
# ---------------------------------------------------------------------------

@app.get("/api/services/{service}/ingest/status")
def ingest_status(service: str):
    """Retorna status da ingestao do servico."""
    return {"status": _ingest_status.get(service, "idle")}


def _trigger_ingest(service: str, config: ServiceConfig):
    """Dispara ingestao em background thread."""
    if _ingest_status.get(service) == "running":
        return  # ja esta rodando

    def _run():
        _ingest_status[service] = "running"
        try:
            run_ingest(config, force=True)
            # Invalida cache do pipeline para recarregar o indice
            _pipeline_cache.pop(service, None)
            _ingest_status[service] = "done"
        except Exception as e:
            _ingest_status[service] = f"error:{e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@app.post("/api/services/{service}/ingest")
def ingest(service: str, force: bool = False):
    """Executa ingestao de documentos."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    try:
        result = run_ingest(config, force=force)
        if result is None:
            return IngestResponse(status="skipped", message="Documentos nao mudaram.")
        return IngestResponse(status="ok", message="Ingestao concluida.")
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Chat REST
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Envia pergunta e recebe resposta completa."""
    try:
        config = ServiceConfig.load(req.service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{req.service}' nao encontrado.")

    pipeline = get_pipeline(config)
    result = pipeline.invoke(req.question)

    fontes = list(set(c.metadata.get("source", "?") for c in result.chunks))

    return ChatResponse(
        resposta=result.resposta,
        fontes=fontes,
        raw_count=result.raw_count,
        rerank_count=result.rerank_count,
    )


# ---------------------------------------------------------------------------
# WebSocket — chat com status + filtro de fontes
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """WebSocket para chat com atualizacao de status em tempo real."""
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            service_name = data.get("service", "default")
            question = data.get("question", "")
            fontes_selecionadas = data.get("fontes_selecionadas", [])

            if not question:
                await ws.send_json({"type": "error", "message": "Pergunta vazia."})
                continue

            try:
                config = ServiceConfig.load(service_name)
            except FileNotFoundError:
                await ws.send_json({"type": "error", "message": f"Servico '{service_name}' nao encontrado."})
                continue

            pipeline = get_pipeline(config)

            status = ["Pensando..."]
            last_status = ""

            loop = asyncio.get_event_loop()

            async def send_status_updates():
                nonlocal last_status
                while True:
                    if status[0] != last_status:
                        last_status = status[0]
                        await ws.send_json({"type": "status", "message": last_status})
                    await asyncio.sleep(0.3)

            status_task = asyncio.create_task(send_status_updates())

            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: pipeline.invoke(
                        question,
                        status=status,
                        fontes_selecionadas=fontes_selecionadas,
                    ),
                )
            except Exception as e:
                status_task.cancel()
                await ws.send_json({
                    "type": "error",
                    "message": str(e),
                    "llm_provider": pipeline.active_provider,
                    "llm_model": pipeline.active_model,
                })
                continue
            finally:
                status_task.cancel()

            fontes = list(set(c.metadata.get("source", "?") for c in result.chunks))

            await ws.send_json({
                "type": "response",
                "resposta": result.resposta,
                "fontes": fontes,
                "raw_count": result.raw_count,
                "rerank_count": result.rerank_count,
                "llm_provider": pipeline.active_provider,
                "llm_model": pipeline.active_model,
            })

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Frontend estatico
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse("web/index.html")

app.mount("/static", StaticFiles(directory="web"), name="static")
