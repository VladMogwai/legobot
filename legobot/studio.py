"""Файл .io для Studio и его открытие.

.io — zip: model.ldr + .info. Studio не объявляет типы документов в Info.plist, поэтому
`open`/двойной клик отвечают «cannot open files in the Stud.io Format». Зато при старте
Studio берёт файлы из аргументов командной строки — так и открываем (каждый раз новое окно).
"""
import json
import subprocess
import zipfile
from pathlib import Path

STUDIO_BINARY = "/Applications/Studio 2.0/Studio.app/Contents/MacOS/Studio"
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
    """Каждый запуск — новое окно Studio; окна, открытые этим же способом раньше, закрываем,
    иначе легко смотреть на старую версию модели."""
    subprocess.run(["pkill", "-f", f"^{STUDIO_BINARY} .*\\.io$"], check=False)
    subprocess.Popen([STUDIO_BINARY, str(Path(io_path).resolve())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
