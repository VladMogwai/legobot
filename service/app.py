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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from legobot.instructions import bill_of_materials, split_steps, write_bom, write_pdf
from legobot.ldraw import write_ldr
from legobot.mosaic import standing_bricks
from legobot.pack import pack_model
from legobot.pixelart import mosaic_from_photo, write_check
from legobot.studio import write_io

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
    job = Job(uuid.uuid4().hex[:12])
    folder = WORK_DIR / job.id
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(photo.filename or "photo.jpg").suffix.lower() or ".jpg"
    (folder / f"photo{suffix}").write_bytes(data)
    with lock:
        jobs[job.id] = job
    executor.submit(_run, job, folder / f"photo{suffix}")
    return {"id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "нет такой задачи")
    out = {"id": job.id, "status": job.status, "error": job.error, "summary": job.summary}
    if job.status == "done":
        out["files"] = {name: f"/jobs/{job.id}/files/{name}" for name in FILES}
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
    job.status = "running"
    folder = photo.parent
    try:
        result = mosaic_from_photo(str(photo))
        bricks = standing_bricks(result.mosaic)
        steps = split_steps(bricks)
        ldr = folder / "model.ldr"
        write_ldr(bricks, str(ldr), "legobot", steps=steps)
        write_io(str(ldr), str(folder / "model.io"))
        (folder / "model.mpd").write_text(pack_model(ldr.read_text()))
        bom = bill_of_materials(bricks)
        write_bom(bom, str(folder / "parts.csv"))
        write_pdf(bricks, steps, str(folder / "instructions.pdf"), "legobot")
        accuracy = write_check(result, str(folder / "check.png"))
        m = result.mosaic
        job.summary = {
            "pixels": [m.width, m.height], "parts": len(bricks), "steps": len(steps),
            "size_cm": [round(m.width * 0.8, 1), round(max(b.z + b.length for b in bricks) * 0.8, 1), round(m.height * 0.96, 1)],
            "colors": [[l.color_name, l.quantity] for l in _color_totals(bom)],
            "accuracy": accuracy,
        }
        job.status = "done"
        log.info("job %s done: %s parts", job.id, len(bricks))
    except Exception as e:  # noqa: BLE001 — пользователю нужна причина, какой бы она ни была
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)
        shutil.rmtree(folder, ignore_errors=True)


def _color_totals(bom):
    from collections import Counter
    totals = Counter()
    for l in bom:
        totals[l.color_name] += l.quantity
    from legobot.instructions import PartLine
    return [PartLine("", "", 0, name, n) for name, n in totals.most_common()]
