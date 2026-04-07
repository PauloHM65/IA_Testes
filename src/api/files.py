"""Router: upload, listagem e remocao de arquivos."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas import FileInfo
from src.api.ingest import trigger_ingest
from src.config import ServiceConfig

router = APIRouter(prefix="/api/services", tags=["files"])


@router.get("/{service}/files")
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


@router.post("/{service}/upload")
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

    trigger_ingest(service, config)

    return {"status": "ok", "file": file.filename, "path": str(dest)}


@router.delete("/{service}/files/{filename}")
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
    trigger_ingest(service, config)

    return {"status": "ok", "message": f"'{filename}' removido."}
