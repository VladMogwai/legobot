"""Самодостаточный MPD для вьюшки: модель плюс все файлы деталей, которые она использует,
и определения цветов. Такой файл three.js LDrawLoader открывает без библиотеки LDraw.

Файлы деталей берутся из vendor/ldraw (подмножество библиотеки в репозитории),
при отсутствии — из установленного Studio.
"""
from pathlib import Path

from .colors import _rows
from .parts import LIBRARY

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "ldraw"
SEARCH = ("parts", "p", "UnOfficial/parts", "UnOfficial/p", "parts/s", "p/48", "UnOfficial/parts/s")


def pack_model(ldr_text: str) -> str:
    """Текст .ldr модели -> текст MPD с цветами и вложенными деталями."""
    main = _normalized(ldr_text)
    used_colors = sorted({int(t[1]) for t in (l.split() for l in main) if len(t) >= 15 and t[0] == "1"})
    colors = [f"0 !COLOUR {r['name']} CODE {r['code']} VALUE #{r['studio_rgb']} EDGE #333333"
              for r in _rows() if int(r["code"]) in used_colors]
    out = ["0 FILE model.ldr", "0 legobot model", *colors, *main]
    done: set[str] = set()
    queue = [ref for ref in _refs(main)]
    while queue:
        ref = queue.pop()
        if ref in done:
            continue
        done.add(ref)
        text = _read(ref)
        if text is None:
            continue
        body = _normalized(text)
        out += ["0 NOFILE", f"0 FILE {_loader_name(ref)}", *body]
        queue += _refs(body)
    out.append("0 NOFILE")
    return "\n".join(out) + "\n"


def _normalized(text: str) -> list[str]:
    """Строки файла без заголовков MPD; ссылки на подфайлы — в нижнем регистре с прямыми слэшами,
    чтобы совпадать с именами вложенных `0 FILE`."""
    out = []
    for line in text.splitlines():
        if line.startswith(("0 FILE", "0 NOFILE")):
            continue
        t = line.split()
        if len(t) >= 15 and t[0] == "1":
            line = " ".join(t[:14] + [" ".join(t[14:]).replace("\\", "/").lower()])
        out.append(line)
    return out


def _loader_name(ref: str) -> str:
    """Имя вложенного файла, под которым его ищет three.js LDrawLoader: ссылки `s/…` он
    переписывает в `parts/s/…`, `48/…` — в `p/48/…`."""
    if ref.startswith("s/"):
        return "parts/" + ref
    if ref.startswith("48/"):
        return "p/" + ref
    return ref


def _refs(lines: list[str]) -> list[str]:
    return [" ".join(t[14:]).replace("\\", "/").lower() for t in (l.split() for l in lines) if len(t) >= 15 and t[0] == "1"]


def _read(ref: str) -> str | None:
    for root in (VENDOR, LIBRARY):
        for folder in SEARCH:
            path = root / folder / ref
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")
    return None
