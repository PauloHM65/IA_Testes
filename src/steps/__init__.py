"""Registry de steps: auto-registro via decorator @register."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.steps.base import BaseStep

STEP_REGISTRY: dict[str, type[BaseStep]] = {}


def register(cls: type[BaseStep]) -> type[BaseStep]:
    """Decorator que registra um step no STEP_REGISTRY pelo seu `name`."""
    if cls.name:
        STEP_REGISTRY[cls.name] = cls
    return cls


# Importa todos os modulos para que o @register rode no import
def _load_all_steps():
    """Importa todos os steps para popular o STEP_REGISTRY."""
    from src.steps import (  # noqa: F401
        retrievers,
        rerankers,
        generators,
        post_processors,
        neighbors,
        classificar,
        buscar_exercicio,
        buscar_materia,
    )
