"""Чтение .ldr обратно в кирпичи по каталогу деталей: (x, z, layer, w, l, color)."""
from legobot.finish import TILES
from legobot.parts import VOCABULARIES

_PARTS = {p.number: p for v in VOCABULARIES.values() for p in v.parts} | {t.number: t for t in TILES.values()}


def load_bricks(path):
    """Кирпичи из каталога; чужие детали (колёса и т.п.) пропускаются — их отдаёт load_fixtures."""
    bricks, height = [], None
    for line in open(path):
        if not line.startswith("1 "):
            continue
        t = line.split()
        color = int(t[1]); cx, cy, cz = map(float, t[2:5]); rot = list(map(float, t[5:14]))
        part = _PARTS.get(t[14][:-4])
        if part is None:
            continue
        height = part.height
        w, l = (part.length, part.width) if rot[0] == 0 else (part.width, part.length)
        bricks.append((int(round(cx / 20 - w / 2)), int(round(cz / 20 - l / 2)), int(round(-cy / height)), w, l, color))
    return bricks, height


def load_fixtures(path):
    """Детали не из каталога: (part, color, (x, y, z) LDU, rot)."""
    out = []
    for line in open(path):
        t = line.split()
        if line.startswith("1 ") and t[14][:-4] not in _PARTS:
            out.append((t[14][:-4], int(t[1]), tuple(map(float, t[2:5])), tuple(map(float, t[5:14]))))
    return out
