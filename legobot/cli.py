"""python -m legobot <фото или 3D-файл> [--parts N | --height CM | --grid N] [--wheels] [--open]"""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path

from .colors import load_palette
from .fixtures import WHEEL_BY_PART
from .ldraw import write_ldr
from .parts import VOCABULARIES
from .pipeline import Build, build
from .sizing import fit_height, fit_parts
from .studio import open_in_studio, write_io

YELLOW = 14
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OUT_DIR = Path("out")


def main() -> None:
    ap = argparse.ArgumentParser(prog="legobot")
    ap.add_argument("input", help="фото (jpg/png/webp) или 3D-файл (STL/OBJ/PLY/GLB, ось Z — вертикаль)")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--grid", type=int, help="штырьков по длинной стороне")
    size.add_argument("--parts", type=int, help="желаемое число деталей")
    size.add_argument("--height", type=float, help="желаемая высота, см")
    ap.add_argument("--unit", choices=VOCABULARIES, default="plates", help="из чего класть: plates (точнее) или bricks")
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw, если у модели нет своего цвета")
    ap.add_argument("--colors", type=int, default=4, help="до скольких цветов сводить цвет модели")
    ap.add_argument("--symmetric", action="store_true", help="зеркальная кладка, даже если меш кривоват (фото)")
    ap.add_argument("--wheels", nargs="?", const="auto", choices=["auto", *WHEEL_BY_PART],
                    help="найти арки и поставить колёса: auto — подобрать по арке, или номер детали")
    ap.add_argument("--no-tiles", action="store_true", help="не заменять верхние пластины тайлами")
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536], help="детализация нейросети для фото")
    ap.add_argument("-o", "--out", help="куда писать .io (рядом ляжет .ldr)")
    ap.add_argument("--open", action="store_true", help="открыть результат в Studio")
    args = ap.parse_args()

    mesh_path = _mesh_from_input(args.input, args.resolution)

    vocabulary = VOCABULARIES[args.unit]
    build_at = partial(build, mesh_path, default_color=args.color, vocabulary=vocabulary,
                       wheels=args.wheels, tiles=not args.no_tiles, max_colors=args.colors,
                       force_symmetric=args.symmetric)
    if args.parts:
        result = fit_parts(build_at, args.parts)
    elif args.height:
        result = fit_height(build_at, mesh_path, args.height, vocabulary)
    else:
        result = build_at(args.grid or 30)

    io_path = Path(args.out) if args.out else _model_dir(args.input) / f"{Path(args.input).stem}_g{result.grid}.io"
    io_path.parent.mkdir(parents=True, exist_ok=True)
    ldr_path = io_path.with_suffix(".ldr")
    write_ldr(result.bricks, str(ldr_path), io_path.stem, result.fixtures)
    write_io(str(ldr_path), str(io_path))
    print(_summary(result))
    print("->", io_path)
    if args.open:
        open_in_studio(str(io_path))


def _model_dir(input_path: str) -> Path:
    """Каждая модель — своя папка: out/<имя>/ с мешем, .ldr, .io и рендерами."""
    d = OUT_DIR / Path(input_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mesh_from_input(path: str, resolution: int) -> str:
    """Фото прогоняется через нейросеть один раз; результат лежит в out/<имя>/<имя>.ply."""
    src = Path(path)
    if src.suffix.lower() not in IMAGE_SUFFIXES:
        return str(src)
    cached = _model_dir(path) / f"{src.stem}.ply"
    if cached.exists():
        print(f"меш из фото уже есть: {cached}")
        return str(cached)
    from .photo import mesh_from_photo  # импорт здесь: gradio_client нужен только для фото
    print("фото → 3D через TRELLIS.2, обычно 1–3 минуты…")
    return mesh_from_photo(str(src), str(cached), resolution=resolution)


def _summary(r: Build) -> str:
    names = {c.code: c.name for c in load_palette()}
    w, l, h = r.size_mm
    parts = Counter(b.part.label for b in r.bricks)
    return (
        f"сетка {r.grid}, {r.vocabulary.name}, {'зеркальная кладка' if r.mirrored else 'модель несимметрична'}\n"
        f"размер {w / 10:.1f} x {l / 10:.1f} x {h / 10:.1f} см, слоёв {len({b.layer for b in r.bricks})}\n"
        f"деталей {r.part_count}: " + ", ".join(f"{k} x{n}" for k, n in sorted(parts.items()))
        + (f" + {len(r.fixtures)} фикс. ({len(r.fixtures) // 3} колёс)" if r.fixtures else "") + "\n"
        f"цвета: " + ", ".join(f"{names.get(c, c)} x{n}" for c, n in r.colors.most_common())
    )


if __name__ == "__main__":
    main()
