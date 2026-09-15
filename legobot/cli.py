"""python -m legobot <фото или 3D-файл> [--parts N | --height CM | --grid N] [--wheels] [--open]"""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path

from .colors import code_by_name, load_palette
from .fixtures import WHEEL_BY_PART
from .ldraw import write_ldr
from .parts import VOCABULARIES
from .pipeline import Build, build
from .sizing import fit_height, fit_parts, fit_size
from .studio import open_in_studio, write_io

YELLOW = 14
DEFAULT_PARTS = 400          # ориентир: модели на 300–500 деталей
ESTIMATE_PARTS = [200, 300, 400, 500, 700, 1000]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OUT_DIR = Path("out")


def main() -> None:
    ap = argparse.ArgumentParser(prog="legobot")
    ap.add_argument("input", help="фото (jpg/png/webp) или 3D-файл (STL/OBJ/PLY/GLB, ось Z — вертикаль)")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--parts", type=int, help="желаемое число деталей (по умолчанию 400)")
    size.add_argument("--size", type=float, help="длина самой длинной стороны, см")
    size.add_argument("--height", type=float, help="желаемая высота, см")
    size.add_argument("--grid", type=int, help="штырьков по длинной стороне (низкоуровневый параметр)")
    ap.add_argument("--estimate", action="store_true", help="не собирать, а показать таблицу «деталей → размер»")
    ap.add_argument("--unit", choices=VOCABULARIES, default="bricks", help="из чего класть: bricks (объёмные фигуры) или plates (низкие формы, машины; втрое точнее по высоте, но деталей больше)")
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw, если у модели нет своего цвета")
    ap.add_argument("--colors", type=int, default=4, help="до скольких цветов сводить цвет модели")
    ap.add_argument("--symmetric", action="store_true", help="зеркальная кладка, даже если меш кривоват (фото)")
    ap.add_argument("--recolor", action="append", default=[], metavar="OLD=NEW",
                    help="заменить подобранный цвет: --recolor Dark_Red=Orange (имена из LDConfig или коды)")
    ap.add_argument("--wheels", nargs="?", const="auto", choices=["auto", *WHEEL_BY_PART],
                    help="найти арки и поставить колёса: auto — подобрать по арке, или номер детали")
    ap.add_argument("--no-tiles", action="store_true", help="не заменять верхние пластины тайлами")
    ap.add_argument("--mosaic", action="store_true", help="плоская пиксельная фигура: один пиксель = один тайл 1x1")
    ap.add_argument("--pixels", type=int, help="для --mosaic: пикселей по ширине, если сетка не находится сама")
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536], help="детализация нейросети для фото")
    ap.add_argument("--backend", choices=["hf", "kaggle"], default="hf", help="где считать фото→3D: HF Space (быстро, квота) или Kaggle (пачкой)")
    ap.add_argument("--variants", type=int, default=1, help="сколько вариаций (seed) сгенерировать из фото")
    ap.add_argument("-o", "--out", help="куда писать .io (рядом ляжет .ldr)")
    ap.add_argument("--open", action="store_true", help="открыть результат в Studio")
    args = ap.parse_args()

    mesh_paths = _meshes_from_input(args.input, args.resolution, args.backend, args.variants)

    vocabulary = VOCABULARIES[args.unit]
    for variant, mesh_path in mesh_paths.items():
        if args.mosaic:
            _build_mosaic(args, mesh_path, variant)
        else:
            _build_one(args, mesh_path, variant, vocabulary)


def _build_mosaic(args, mesh_path: str, variant: str) -> None:
    from .mosaic import mosaic_bricks, mosaic_from_mesh
    mosaic = mosaic_from_mesh(mesh_path, max_colors=args.colors, pixels_wide=args.pixels)
    bricks = mosaic_bricks(mosaic)
    if args.recolor:
        mapping = {code_by_name(a): code_by_name(b) for a, b in (r.split("=") for r in args.recolor)}
        bricks = [b.__class__(**{**b.__dict__, "color": mapping.get(b.color, b.color)}) for b in bricks]
    suffix = f"_{variant}" if variant else ""
    io_path = Path(args.out) if args.out else _model_dir(args.input) / f"{Path(args.input).stem}{suffix}_mosaic.io"
    io_path.parent.mkdir(parents=True, exist_ok=True)
    ldr_path = io_path.with_suffix(".ldr")
    write_ldr(bricks, str(ldr_path), io_path.stem)
    write_io(str(ldr_path), str(io_path))
    names = {c.code: c.name for c in load_palette(common_only=False)}
    colors = Counter(b.color for b in bricks if b.layer == 1)
    print(f"мозаика {mosaic.width}x{mosaic.height} пикселей, шаг сетки {mosaic.pitch_px:.1f} px растра")
    print(f"деталей {len(bricks)}: тайлов 1x1 {sum(colors.values())}, подложка {len(bricks) - sum(colors.values())}")
    print("цвета: " + ", ".join(f"{names.get(c, c)} x{n}" for c, n in colors.most_common()))
    print("->", io_path)
    if args.open:
        open_in_studio(str(io_path))


def _build_one(args, mesh_path: str, variant: str, vocabulary) -> None:
    build_at = partial(build, mesh_path, default_color=args.color, vocabulary=vocabulary,
                       wheels=args.wheels, tiles=not args.no_tiles, max_colors=args.colors,
                       force_symmetric=args.symmetric,
                       recolor_map={code_by_name(a): code_by_name(b) for a, b in (r.split("=") for r in args.recolor)})
    if args.estimate:
        _print_estimate(build_at)
        return
    if args.size:
        result = fit_size(build_at, args.size)
    elif args.height:
        result = fit_height(build_at, mesh_path, args.height, vocabulary)
    elif args.grid:
        result = build_at(args.grid)
    else:
        result = fit_parts(build_at, args.parts or DEFAULT_PARTS)

    suffix = f"_{variant}" if variant else ""
    io_path = Path(args.out) if args.out else _model_dir(args.input) / f"{Path(args.input).stem}{suffix}_g{result.grid}.io"
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


def _meshes_from_input(path: str, resolution: int, backend: str, variants: int) -> dict[str, str]:
    """{вариант: меш}. Для 3D-файла — он сам. Для фото — по одному мешу на seed, с кэшем в out/<имя>/."""
    src = Path(path)
    if src.suffix.lower() not in IMAGE_SUFFIXES:
        return {"": str(src)}
    seeds = list(range(variants))
    model_dir = _model_dir(path)
    cached = {f"s{seed}": model_dir / f"{src.stem}_s{seed}.ply" for seed in seeds}
    missing = {v: p for v, p in cached.items() if not p.exists()}
    if not missing:
        print("меши из фото уже есть:", ", ".join(str(p) for p in cached.values()))
    elif backend == "kaggle":
        from .kaggle3d import meshes_from_photos
        print(f"фото → 3D на Kaggle, seeds {seeds}; сборка стека 30–60 минут…")
        meshes_from_photos([str(src)], str(model_dir), resolution=resolution, seeds=tuple(seeds))
    else:
        from .photo import mesh_from_photo  # импорт здесь: gradio_client нужен только для фото
        for variant, ply in missing.items():
            print(f"фото → 3D через TRELLIS.2 ({variant}), обычно 1–3 минуты…")
            mesh_from_photo(str(src), str(ply), resolution=resolution, seed=int(variant[1:]))
    return {v: str(p) for v, p in cached.items() if p.exists()}


def _print_estimate(build_at) -> None:
    print(f"{'цель':>6} {'деталей':>8} {'Ш x Д x В, см':>20}  {'из них 1x1':>10}")
    for target in ESTIMATE_PARTS:
        r = fit_parts(build_at, target)
        w, l, h = r.size_mm
        ones = sum(1 for b in r.bricks if b.part.area == 1)
        print(f"{target:6d} {r.part_count:8d} {w / 10:6.1f} x {l / 10:5.1f} x {h / 10:5.1f}  {ones:10d}")
    print("выбери: --parts N (по умолчанию 400), --size СМ или --height СМ")


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
