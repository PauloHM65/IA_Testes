"""Router: chat REST + WebSocket."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.config import ServiceConfig
from src.logging.file_logger import get_logger
from src.logging.performance import get_perf_logger
from src.pipeline import get_pipeline

router = APIRouter(tags=["chat"])

_ws_interaction_count: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Chat REST
# ---------------------------------------------------------------------------

@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Envia pergunta e recebe resposta completa."""
    try:
        config = ServiceConfig.load(req.service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{req.service}' nao encontrado.")

    pipeline = get_pipeline(config)
    result = pipeline.invoke(req.question)

    total_time = result.timings.pop("pipeline_total", 0.0)
    get_perf_logger().log_interaction(
        service=req.service,
        pergunta=req.question,
        timings=result.timings,
        total_time=total_time,
        provider=pipeline.active_provider,
        model=pipeline.active_model,
    )

    fontes = list(set(c.metadata.get("source", "?") for c in result.chunks))

    return ChatResponse(
        resposta=result.resposta,
        fontes=fontes,
        raw_count=result.raw_count,
        rerank_count=result.rerank_count,
    )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws/chat")
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

            # Log da interacao
            _ws_interaction_count[service_name] = _ws_interaction_count.get(service_name, 0) + 1
            get_logger().interacao(
                numero=_ws_interaction_count[service_name],
                pergunta=question,
                chunks=result.chunks,
                contexto=result.contexto,
                resposta=result.resposta,
                generated_queries=result.generated_queries,
                raw_count=result.raw_count,
                rerank_count=result.rerank_count,
            )

            # Log de eficiencia
            total_time = result.timings.pop("pipeline_total", 0.0)
            get_perf_logger().log_interaction(
                service=service_name,
                pergunta=question,
                timings=result.timings,
                total_time=total_time,
                provider=pipeline.active_provider,
                model=pipeline.active_model,
            )

    except WebSocketDisconnect:
        pass
