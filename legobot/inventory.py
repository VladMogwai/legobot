"""Состав любой модели Studio: какие детали, в каких цветах, по каким категориям.

.io — zip с паролем (Studio шифрует их одним известным паролем), внутри model.ldr в формате
LDraw с подмоделями (`0 FILE имя` … `0 NOFILE`). Подмодели раскрываются рекурсивно:
если одна и та же нога вставлена дважды, её детали считаются дважды.
"""
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .colors import LDCONFIG_PATH
from .parts import part_name

IO_PASSWORD = b"soho0909"

# Категория — по началу официального имени детали; порядок важен (первое совпадение).
CATEGORIES = [
    ("Скосы", r"^Slope"),
    ("Technic", r"^Technic"),
    ("Пластины", r"^Plate"),
    ("Тайлы", r"^Tile"),
    ("Кирпичи", r"^Brick"),
    ("Панели и арки", r"^(Panel|Arch|Wedge|Window|Windscreen)"),
    ("Шарниры и крепления", r"^(Hinge|Bar|Clip|Bracket|Turntable)"),
    ("Круглое", r"^(Cylinder|Cone|Dish|Round|Wheel|Tyre)"),
    ("Минифигурки", r"^Minifig"),
]


@dataclass(frozen=True)
class InventoryLine:
    number: str
    name: str
    color: int
    color_name: str
    quantity: int
    category: str


def read_model(path: str) -> str:
    """Текст LDraw из .io (с паролем) или из .ldr/.mpd."""
    p = Path(path)
    if p.suffix.lower() == ".io":
        with zipfile.ZipFile(p) as z:
            return z.read("model.ldr", pwd=IO_PASSWORD).decode("utf-8", errors="ignore")
    return p.read_text(encoding="utf-8", errors="ignore")


def inventory(ldraw_text: str) -> list[InventoryLine]:
    """Список деталей модели с учётом подмоделей."""
    submodels = _split_submodels(ldraw_text)
    root = next(iter(submodels))
    counts: Counter = Counter()
    _count(root, submodels, 1, counts, depth=0)
    names = _color_names()
    lines = []
    for (n, c), q in counts.items():
        name = part_name(n)
        moved = re.match(r"~Moved to (\S+)", name)   # устаревший номер: имя у новой детали
        if moved:
            name = f"{part_name(moved.group(1))} (устар. {n})"
        lines.append(InventoryLine(n, name, c, names.get(c, str(c)), q, _category(name)))
    return sorted(lines, key=lambda l: (-l.quantity, l.name))


def _color_names() -> dict[int, str]:
    """Все цвета LDConfig, включая металлики и прозрачные (в палитре сборки их нет)."""
    names = {}
    for line in open(LDCONFIG_PATH, encoding="utf-8", errors="ignore"):
        m = re.match(r"0 !COLOUR (\S+)\s+CODE\s+(\d+)", line)
        if m:
            names[int(m.group(2))] = m.group(1)
    return names


def _split_submodels(text: str) -> dict[str, list[str]]:
    """{имя файла: строки}; первая запись — главная модель."""
    models: dict[str, list[str]] = {}
    current = "main"
    models[current] = []
    for line in text.splitlines():
        t = line.split(maxsplit=2)
        if len(t) >= 3 and t[0] == "0" and t[1] == "FILE":
            current = t[2].strip().lower()
            models.setdefault(current, [])
        elif not (len(t) >= 2 and t[0] == "0" and t[1] == "NOFILE"):
            models[current].append(line)
    if not models["main"]:
        del models["main"]
    return models


def _count(name: str, submodels: dict, times: int, counts: Counter, depth: int) -> None:
    if depth > 20:
        return
    for line in submodels[name]:
        t = line.split()
        if len(t) < 15 or t[0] != "1":
            continue
        ref = " ".join(t[14:]).lower()
        if ref in submodels:
            _count(ref, submodels, times, counts, depth + 1)
        else:
            counts[(ref.removesuffix(".dat"), int(t[1]))] += times


def _category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.match(pattern, name):
            return label
    return "Прочее"


def report(lines: list[InventoryLine]) -> str:
    total = sum(l.quantity for l in lines)
    by_cat = Counter()
    for l in lines:
        by_cat[l.category] += l.quantity
    out = [f"деталей {total}, типов {len(lines)}, цветов {len({l.color for l in lines})}", "",
           "по категориям: " + ", ".join(f"{c} {n} ({n * 100 // total}%)" for c, n in by_cat.most_common()), ""]
    out += [f"{l.quantity:4d} ×  {l.number:<10} {l.name:<45} {l.color_name}" for l in lines]
    return "\n".join(out)
