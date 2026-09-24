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
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # скрипт запускают и без PYTHONPATH — пакет лежит рядом

from legobot.finish import TILES      # noqa: E402 — после настройки пути
from legobot.parts import BRICKS, PLATES, RETIRED   # noqa: E402

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


def check_geometry() -> None:
    """Все детали словаря и их подфайлы должны лежать в vendor/ldraw: без геометрии модель
    соберётся, но во вьюшке не покажется (LDrawLoader: Subobject could not be loaded)."""
    from legobot.pack import SEARCH, VENDOR
    stack = [f"{n}.dat" for n in ({p.number for v in (BRICKS, PLATES) for p in v.parts}
                                  | {p.number for p in RETIRED} | {t.number for t in TILES.values()})]
    seen, missing = set(), []
    while stack:
        name = stack.pop().replace("\\", "/").lower()
        if name in seen:
            continue
        seen.add(name)
        path = next((VENDOR / folder / name for folder in SEARCH if (VENDOR / folder / name).exists()), None)
        if path is None:
            missing.append(name)
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            t = line.split()
            if len(t) >= 15 and t[0] == "1":
                stack.append(" ".join(t[14:]))
    if missing:
        raise SystemExit(f"нет геометрии в vendor/ldraw: {sorted(missing)[:8]} — запусти tools/vendor_ldraw.py")


def availability_csv() -> str:
    """Список пар деталь+цвет с Pick a Brick без цен: в браузере нужна доступность, а цены
    lego.com мы не публикуем."""
    src = ROOT / "catalog" / "pab.csv"
    if not src.exists():
        return "ldraw,color_code,cents,channel\n"
    with open(src, newline="") as f:
        rows = [(r["ldraw"], r["color_code"]) for r in csv.DictReader(f)]
    return "ldraw,color_code,cents,channel\n" + "".join(f"{n},{c},,\n" for n, c in rows)


def main() -> None:
    check_geometry()
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
        z.writestr("catalog/pab.csv", availability_csv())
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
