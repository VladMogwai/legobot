"""Движок для браузера: python tools/web_engine.py -> web/public/engine/legobot.zip

Архив с пакетом legobot, настройками, каталогом (только детали словаря) и геометрией деталей
(vendor/ldraw) — всё, что нужно `legobot.web.build` в Pyodide. Раскладка внутри архива повторяет
репозиторий (`legobot/`, `catalog/`, `vendor/ldraw/`, `legobot.toml`), потому что модули ищут
данные относительно `__file__`. Pick a Brick (catalog/pab.csv) в архив не кладётся: это данные
lego.com для личного пользования; без него покупаемость оценивается по числу наборов.
"""
import csv
import hashlib
import io
import zipfile
from pathlib import Path

from legobot.finish import TILES
from legobot.parts import BRICKS, PLATES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "public" / "engine"


def vocabulary_numbers() -> set[str]:
    return {p.number for p in BRICKS.parts} | {p.number for p in PLATES.parts} | {t.number for t in TILES.values()}


def trimmed_csv(path: Path, key: str, keep: set[str]) -> str:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r[key] in keep]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=reader.fieldnames)
        w.writeheader(); w.writerows(rows)
        return buf.getvalue()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = vocabulary_numbers()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for py in sorted((ROOT / "legobot").glob("*.py")):
            z.write(py, f"legobot/{py.name}")
        z.write(ROOT / "legobot.toml", "legobot.toml")
        z.writestr("catalog/colors.csv", (ROOT / "catalog" / "colors.csv").read_text())
        z.writestr("catalog/parts.csv", trimmed_csv(ROOT / "catalog" / "parts.csv", "ldraw", numbers))
        z.writestr("catalog/part_colors.csv", trimmed_csv(ROOT / "catalog" / "part_colors.csv", "ldraw", numbers))
        for f in sorted((ROOT / "vendor" / "ldraw").rglob("*")):
            if f.is_file():
                z.write(f, f"vendor/ldraw/{f.relative_to(ROOT / 'vendor' / 'ldraw')}")
    data = buf.getvalue()
    (OUT / "legobot.zip").write_bytes(data)
    stamp = hashlib.sha256(data).hexdigest()[:10]
    (OUT / "version.js").write_text(f'window.LEGOBOT_ENGINE = "{stamp}";\n')
    print(f"-> {OUT / 'legobot.zip'}: {len(data) // 1024} КБ, версия {stamp}")


if __name__ == "__main__":
    main()
