"""Точка входа десктопного бэкенда (PyInstaller): python service/run_server.py [порт]

Данные (каталог, геометрия деталей, страница, модель rembg) лежат рядом с кодом —
в собранном приложении это папка _internal; пути в legobot считаются от корня пакета,
поэтому ничего настраивать не нужно, кроме папки с моделью rembg."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LEGOBOT_LOCAL", "1")
os.environ.setdefault("LEGOBOT_WORK", str(Path.home() / "Library" / "Application Support" / "legobot" / "jobs"))
if (ROOT / "rembg_home").exists():
    os.environ.setdefault("REMBG_HOME", str(ROOT / "rembg_home"))
os.environ.setdefault("MODEL_CHECKSUM_DISABLED", "1")

import uvicorn  # noqa: E402

from service.app import app  # noqa: E402

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
