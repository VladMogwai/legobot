"""Файл .io для Studio и его открытие.

.io — zip: model.ldr + .info. Такой файл Studio открывает двойным кликом,
в отличие от .ldr, который надо импортировать через меню.
"""
import json
import subprocess
import zipfile
from pathlib import Path

STUDIO_APP = "/Applications/Studio 2.0/Studio.app"
STUDIO_VERSION = "2.26.8_1"
PARTS_DB_VERSION = 200


def write_io(ldr_path: str, io_path: str) -> str:
    ldr = Path(ldr_path).read_text()
    total = sum(1 for line in ldr.splitlines() if line.startswith("1 "))
    with zipfile.ZipFile(io_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.ldr", ldr)
        z.writestr(".info", json.dumps({"version": STUDIO_VERSION, "total_parts": total, "parts_db_version": PARTS_DB_VERSION}))
    return io_path


def open_in_studio(io_path: str) -> None:
    subprocess.run(["open", "-a", STUDIO_APP, io_path], check=True)
