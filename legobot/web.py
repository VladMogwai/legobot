"""Точка входа для браузера (Pyodide): картинка в памяти → все файлы модели в памяти.

Тот же путь, что у `legobot --mosaic`, но без диска и без Studio. Зависимости — numpy, scipy,
scikit-image, Pillow и matplotlib (последний приходит вместе с scikit-image): всё это есть в Pyodide. Нейросеть не нужна: rembg (пёстрый фон) в браузере пока нет,
такие картинки просят ровный/прозрачный/шахматный фон.
"""
import io
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .chart import write_chart
from .guide import write_guide
from .colors import code_by_name, load_palette
from .instructions import bill_of_materials, split_steps
from .ldraw import write_ldr
from .mosaic import mosaic_bricks, standing_bricks
from .pack import pack_model
from .pixelart import is_full_frame, mosaic_from_image, mosaic_from_photo, write_check
from .studio import PARTS_DB_VERSION, STUDIO_VERSION


def build(image_bytes: bytes, mode: str = "auto", background: str = "auto", width: int = 0,
          max_colors: int = 0, volume_depth: int = 4, recolor: dict | None = None, contrast: bool = False,
          base: bool = True) -> dict:
    """mode: auto | standing | volume | flat. background: auto | cut | keep. width > 0 — переложить
    любую картинку в пиксель-арт такой ширины. base — класть ли у панно свою подложку из пластин.
    Возвращает {"files": {имя: bytes|str}, "summary": {...}}."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "input.png"
        image_path.write_bytes(image_bytes)
        panel = is_full_frame(np.asarray(Image.open(image_path).convert("RGB")).astype(float) / 255)
        if mode == "auto":                       # фото панно собираем панно, всё остальное — фигуркой
            mode = "flat" if panel else "standing"
        # панно — это вся картинка целиком: фон у него не вырезают, а выкладывают
        keep = mode == "flat" or background == "keep" or (background == "auto" and panel)
        colors = max_colors or (32 if keep else 16)
        if width:
            result = mosaic_from_image(str(image_path), width, max_colors=colors, keep_background=keep, contrast=contrast)
        else:
            result = mosaic_from_photo(str(image_path), max_colors=colors, keep_background=keep)
        mosaic = result.mosaic
        for old, new in (recolor or {}).items():
            mosaic.codes[mosaic.codes == code_by_name(old)] = code_by_name(new)
        if mode == "volume":
            from .inflate import volume_bricks
            bricks = volume_bricks(mosaic, volume_depth)
        elif mode == "flat":
            bricks = mosaic_bricks(mosaic, base=base)
        else:
            bricks = standing_bricks(mosaic)
        out = _outputs(bricks, mosaic, Path(tmp), mode)
        check_path = Path(tmp) / "check.png"
        out["summary"]["accuracy"] = write_check(result, str(check_path))
        out["files"]["check.png"] = check_path.read_bytes()
        return out


def build_from_grid(codes: list[list[int]], mode: str = "standing", volume_depth: int = 4, base: bool = True) -> dict:
    """Пересборка из сетки, отредактированной на странице: codes[x][y], -1 — пусто."""
    from .mosaic import Mosaic
    grid = np.array(codes, dtype=int)
    mosaic = Mosaic(grid, 1.0, *grid.shape)
    if mode == "volume":
        from .inflate import volume_bricks
        bricks = volume_bricks(mosaic, volume_depth)
    elif mode == "flat":
        bricks = mosaic_bricks(mosaic, base=base)
    else:
        bricks = standing_bricks(mosaic)
    with tempfile.TemporaryDirectory() as tmp:
        return _outputs(bricks, mosaic, Path(tmp), mode)


def _outputs(bricks, mosaic, tmp: Path, mode: str) -> dict:
    steps = split_steps(bricks)
    ldr_path = tmp / "model.ldr"
    write_ldr(bricks, str(ldr_path), "legobot", steps=steps)
    ldr = ldr_path.read_text()
    files = {"model.ldr": ldr, "model.io": _io_bytes(ldr), "model.mpd": pack_model(ldr)}
    bom = bill_of_materials(bricks)
    files["parts.csv"] = "part,name,color_code,color,quantity\n" + "".join(
        f"{l.number},{l.name},{l.color},{l.color_name},{l.quantity}\n" for l in bom)
    kind = {"standing": "стоячая", "volume": "объёмная", "flat": "панно"}[mode]
    if mode == "flat":
        chart_path = tmp / "chart.png"
        write_chart(bricks, str(chart_path))
        files["chart.png"] = chart_path.read_bytes()
    guide_path = tmp / "instructions.pdf"
    guide_pages = write_guide(bricks, str(guide_path), "legobot", kind)
    files["instructions.pdf"] = guide_path.read_bytes()
    names = {c.code: c.name for c in load_palette(common_only=False)}
    width_studs = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
    depth = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
    height = (max(b.layer for b in bricks) + 1) * bricks[0].part.height / 20
    summary = {
        "kind": kind, "pixels": [mosaic.width, mosaic.height], "parts": len(bricks),
        "steps": len(steps), "pages": guide_pages,
        "size_cm": [round(width_studs * 0.8, 1), round(depth * 0.8, 1), round(height * 0.8, 1)],
        "colors": [[names.get(c, str(c)), n] for c, n in Counter(b.color for b in bricks).most_common()],
        "grid": mosaic.codes.tolist(), "price_usd": _price(bricks), "palette": _palette(),
    }
    return {"files": files, "summary": summary}


def _io_bytes(ldr: str) -> bytes:
    buf = io.BytesIO()
    total = sum(1 for line in ldr.splitlines() if line.startswith("1 "))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.ldr", ldr)
        z.writestr(".info", json.dumps({"version": STUDIO_VERSION, "total_parts": total, "parts_db_version": PARTS_DB_VERSION}))
    return buf.getvalue()


def _palette() -> list[int]:
    """Цвета, которыми бот красит модель: ходовые и продающиеся — ими же красит редактор."""
    from . import catalog
    purchasable = catalog.purchasable_colors()
    return [c.code for c in load_palette(common_only=True) if purchasable is None or c.code in purchasable]


def _price(bricks) -> float | None:
    from .catalog import price_cents
    prices = [price_cents(b.part.number, b.color) for b in bricks]
    return None if all(p is None for p in prices) else round(sum(p or 0 for p in prices) / 100, 2)
