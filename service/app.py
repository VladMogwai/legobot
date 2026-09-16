"""HTTP-сервис: фото пиксельной фигурки -> модель LEGO (.io для Studio, MPD для вьюшки,
PDF-инструкция, список деталей, картинка самопроверки).

Задачи считаются по одной в фоновом потоке (rembg + numpy, ~40 с на фото), результаты лежат
на диске в WORK_DIR. Состояние — в памяти: для одного процесса на Space этого достаточно.
"""
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from legobot.instructions import bill_of_materials, split_steps, write_bom, write_pdf
from legobot.inventory import read_model
from legobot.ldraw import read_bricks, write_ldr
from legobot.mosaic import Mosaic, standing_bricks
from legobot.pack import pack_model
from legobot.parts import VOCABULARIES
from legobot.pipeline import build
from legobot.pixelart import mosaic_from_photo, write_check
from legobot.sizing import fit_parts
from legobot.studio import write_io

VOLUME_PARTS = 400   # бюджет деталей объёмной фигурки

WORK_DIR = Path("/tmp/legobot")
MAX_UPLOAD = 15 * 1024 * 1024
FILES = {"model.io": "application/octet-stream", "model.mpd": "text/plain", "model.ldr": "text/plain",
         "instructions.pdf": "application/pdf", "parts.csv": "text/csv", "check.png": "image/png"}


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    error: str | None = None
    summary: dict = field(default_factory=dict)


log = logging.getLogger("legobot")
app = FastAPI(title="legobot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, Job] = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)


@app.post("/jobs")
async def create_job(photo: UploadFile = File(...)) -> dict:
    data = await photo.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "файл больше 15 МБ")
    job = _new_job()
    suffix = Path(photo.filename or "photo.jpg").suffix.lower() or ".jpg"
    photo_path = WORK_DIR / job.id / f"photo{suffix}"
    photo_path.write_bytes(data)
    executor.submit(_run, job, photo_path)
    return {"id": job.id, "status": job.status}


def _new_job() -> Job:
    job = Job(uuid.uuid4().hex[:12])
    (WORK_DIR / job.id).mkdir(parents=True, exist_ok=True)
    with lock:
        jobs[job.id] = job
    return job


@app.post("/grids")
def build_from_grid(codes: list[list[int]] = Body(..., embed=True)) -> dict:
    """Сборка из сетки пикселей (коды LDraw по столбцам, -1 — пусто) — например, отредактированной."""
    grid = np.array(codes, dtype=int)
    if grid.ndim != 2 or grid.size == 0 or grid.size > 200 * 200:
        raise HTTPException(400, "сетка должна быть прямоугольной и не больше 200×200")
    job = _new_job()
    executor.submit(_run_mosaic, job, Mosaic(grid, 0.0, *grid.shape), None)
    return {"id": job.id, "status": job.status}


@app.post("/models")
async def upload_model(model: UploadFile = File(...)) -> dict:
    """Готовая модель (.io из Studio или .ldr) → инструкция, список деталей, превью."""
    data = await model.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "файл больше 15 МБ")
    job = _new_job()
    suffix = Path(model.filename or "model.io").suffix.lower()
    src = WORK_DIR / job.id / f"upload{suffix}"
    src.write_bytes(data)
    executor.submit(_run_model, job, src)
    return {"id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "нет такой задачи")
    out = {"id": job.id, "status": job.status, "error": job.error, "summary": job.summary}
    if job.status == "done":
        out["files"] = {name: f"/jobs/{job.id}/files/{name}" for name in FILES if (WORK_DIR / job.id / name).exists()}
    return out


@app.get("/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    path = WORK_DIR / job_id / name
    if name not in FILES or not path.exists():
        raise HTTPException(404, "нет такого файла")
    return FileResponse(path, media_type=FILES[name], filename=name)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "jobs": len(jobs)}


def _run(job: Job, photo: Path) -> None:
    """Пиксельная фигурка, если на фото есть сетка пикселей; иначе объёмная через фото→3D."""
    job.status = "running"
    try:
        result = mosaic_from_photo(str(photo))
    except ValueError as e:                    # сетки нет — это не пиксель-арт
        log.info("job %s: %s — объёмный путь", job.id, e)
        _run_volume(job, photo)
        return
    except Exception as e:  # noqa: BLE001 — пользователю нужна причина, какой бы она ни была
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)
        return
    _run_mosaic(job, result.mosaic, result)


def _run_volume(job: Job, photo: Path) -> None:
    """Объёмная фигурка: фото → 3D на HF Space (TRELLIS.2, бесплатная GPU-квота) → пластины."""
    from legobot.photo import hf_token, mesh_from_photo
    folder = WORK_DIR / job.id
    job.summary = {"stage": "фото → 3D (1–3 минуты)"}
    try:
        if not hf_token():
            raise RuntimeError("объёмные фигурки недоступны: на сервере нет токена Hugging Face (HF_TOKEN)")
        mesh = mesh_from_photo(str(photo), str(folder / "mesh.ply"))
        job.summary = {"stage": "укладка деталей"}
        vocabulary = VOCABULARIES["plates"]
        result = fit_parts(lambda grid: build(mesh, grid, 14, vocabulary, force_symmetric=True), VOLUME_PARTS)
        _write_outputs(job, folder, result.bricks)
        job.summary["kind"] = "объёмная"
        job.status = "done"
        log.info("job %s done (volume): %s parts", job.id, len(result.bricks))
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)


def _run_mosaic(job: Job, mosaic: Mosaic, photo_result) -> None:
    """Сборка стоячей фигурки из сетки; photo_result — для картинки самопроверки."""
    job.status = "running"
    folder = WORK_DIR / job.id
    try:
        bricks = standing_bricks(mosaic)
        _write_outputs(job, folder, bricks)
        if photo_result is not None:
            job.summary["accuracy"] = write_check(photo_result, str(folder / "check.png"))
        job.summary["pixels"] = [mosaic.width, mosaic.height]
        job.summary["grid"] = mosaic.codes.tolist()
        job.summary["kind"] = "пиксельная"
        job.status = "done"
        log.info("job %s done: %s parts", job.id, len(bricks))
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)


def _run_model(job: Job, src: Path) -> None:
    """Инструкция и превью для модели, собранной или отредактированной в Studio."""
    job.status = "running"
    folder = WORK_DIR / job.id
    try:
        bricks, skipped = read_bricks(read_model(str(src)))
        if not bricks:
            raise ValueError("в файле нет знакомых деталей (кирпичи, пластины, тайлы)")
        _write_outputs(job, folder, bricks)
        if skipped:
            job.summary["skipped"] = skipped
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)


def _write_outputs(job: Job, folder: Path, bricks) -> None:
    steps = split_steps(bricks)
    ldr = folder / "model.ldr"
    write_ldr(bricks, str(ldr), "legobot", steps=steps)
    write_io(str(ldr), str(folder / "model.io"))
    (folder / "model.mpd").write_text(pack_model(ldr.read_text()))
    bom = bill_of_materials(bricks)
    write_bom(bom, str(folder / "parts.csv"))
    write_pdf(bricks, steps, str(folder / "instructions.pdf"), "legobot")
    width = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
    depth = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
    height = (max(b.layer for b in bricks) + 1) * bricks[0].part.height / 20
    job.summary = {
        "parts": len(bricks), "steps": len(steps),
        "size_cm": [round(width * 0.8, 1), round(depth * 0.8, 1), round(height * 0.8, 1)],
        "colors": [[l.color_name, l.quantity] for l in _color_totals(bom)],
    }


def _color_totals(bom):
    from collections import Counter
    totals = Counter()
    for l in bom:
        totals[l.color_name] += l.quantity
    from legobot.instructions import PartLine
    return [PartLine("", "", 0, name, n) for name, n in totals.most_common()]
