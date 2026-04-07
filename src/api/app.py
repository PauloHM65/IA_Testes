"""App FastAPI: monta routers e middleware."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

from src.config import ServiceConfig, list_services, get_default_service
from src.pipeline import get_pipeline
from src.api.ingest import init_ingest_status
from src.api import services as services_router
from src.api import files as files_router
from src.api import ingest as ingest_router
from src.api import chat as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc_names = list_services()
    init_ingest_status(svc_names)
    if svc_names:
        default = get_default_service()
        try:
            get_pipeline(ServiceConfig.load(default))
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

# Routers
app.include_router(services_router.router)
app.include_router(files_router.router)
app.include_router(ingest_router.router)
app.include_router(chat_router.router)


# Frontend estatico
@app.get("/")
def serve_index():
    return FileResponse("web/index.html")


app.mount("/static", StaticFiles(directory="web"), name="static")
