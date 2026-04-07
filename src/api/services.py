"""Router: listagem de servicos e modelos."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelSwitchRequest
from src.config import ServiceConfig, list_services
from src.pipeline import get_pipeline

router = APIRouter(prefix="/api/services", tags=["services"])


@router.get("")
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


@router.get("/{service}/models")
def get_available_models(service: str):
    """Retorna modelos disponiveis e o ativo."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    pipeline = get_pipeline(config)
    return {
        "active_provider": pipeline.active_provider,
        "active_model": pipeline.active_model,
        "available": list(config.available_models),
    }


@router.post("/{service}/models/switch")
def switch_model(service: str, req: ModelSwitchRequest):
    """Troca o modelo ativo do pipeline."""
    try:
        config = ServiceConfig.load(service)
    except FileNotFoundError:
        raise HTTPException(404, f"Servico '{service}' nao encontrado.")

    allowed = [(m["provider"], m["model"]) for m in config.available_models]
    if (req.provider, req.model) not in allowed:
        raise HTTPException(400, f"Modelo '{req.provider}/{req.model}' nao permitido.")

    pipeline = get_pipeline(config)
    pipeline.switch_model(req.provider, req.model)
    return {
        "status": "ok",
        "active_provider": pipeline.active_provider,
        "active_model": pipeline.active_model,
    }
