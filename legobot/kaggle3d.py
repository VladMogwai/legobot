"""Фото -> 3D через TRELLIS.2 на Kaggle (бесплатные 30 GPU-часов в неделю).

Фото загружаются новой версией датасета legobot-photos, ноутбук legobot-trellis
собирает TRELLIS.2 и гонит все фото подряд, GLB забираются обратно.
Сборка расширений занимает 20-40 минут, поэтому фото выгоднее отправлять пачкой.
Токен Kaggle — KAGGLE_TOKEN в .env.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .photo import ENV_FILE, _load_z_up

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "kaggle" / "kernel"
PHOTOS_DIR = ROOT / "kaggle" / "photos"
KERNEL = "vladmogwai/legobot-trellis"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def kaggle_token() -> str | None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return os.environ["KAGGLE_API_TOKEN"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("KAGGLE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _kaggle(*args: str) -> str:
    env = {**os.environ, "KAGGLE_API_TOKEN": kaggle_token() or ""}
    return subprocess.run([KAGGLE, *args], env=env, capture_output=True, text=True, check=False).stdout.strip()


def meshes_from_photos(image_paths: list[str], out_dir: str, resolution: int = 1024, seeds: tuple[int, ...] = (0,),
                       poll_seconds: int = 120, timeout_minutes: int = 180) -> dict[str, str]:
    """Возвращает {имя_s<seed>: путь к PLY}. Блокирует до конца прогона на Kaggle."""
    for old in PHOTOS_DIR.glob("*.*"):
        if old.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".json") and old.name != "dataset-metadata.json":
            old.unlink()
    for p in image_paths:
        shutil.copy(p, PHOTOS_DIR / Path(p).name)
    (PHOTOS_DIR / "config.json").write_text(json.dumps({"seeds": list(seeds), "resolution": str(resolution)}))
    print("kaggle: загружаю фото…", flush=True)
    print(_kaggle("datasets", "version", "-p", str(PHOTOS_DIR), "-m", "photos", "--dir-mode", "zip"))
    print("kaggle: запускаю ноутбук…", flush=True)
    print(_kaggle("kernels", "push", "-p", str(KERNEL_DIR)))

    started = time.time()
    while True:
        status = _kaggle("kernels", "status", KERNEL)
        print(f"kaggle: {status}", flush=True)
        if "complete" in status.lower() or "error" in status.lower() or "cancel" in status.lower():
            break
        if time.time() - started > timeout_minutes * 60:
            raise TimeoutError("Kaggle: прогон не завершился вовремя")
        time.sleep(poll_seconds)

    raw = Path(out_dir) / "kaggle"
    raw.mkdir(parents=True, exist_ok=True)
    print(_kaggle("kernels", "output", KERNEL, "-p", str(raw)))
    results = {}
    for glb in sorted(raw.glob("**/*.glb")):
        ply = Path(out_dir) / f"{glb.stem}.ply"
        _load_z_up(str(glb)).export(str(ply))
        results[glb.stem] = str(ply)
    return results
