"""python -m legobot <mesh> (--grid N | --parts N | --height CM) [--color C] [-o out.ldr]"""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path

from .colors import load_palette
from .fixtures import WHEEL_BY_PART
from .parts import VOCABULARIES
from .ldraw import write_ldr
from .pipeline import Build, build
from .sizing import fit_height, fit_parts

YELLOW = 14


def main() -> None:
    ap = argparse.ArgumentParser(prog="legobot")
    ap.add_argument("mesh", help="STL/OBJ/PLY/GLB, ось Z — вертикаль")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--grid", type=int, help="штырьков по длинной стороне")
    size.add_argument("--parts", type=int, help="желаемое число деталей")
    size.add_argument("--height", type=float, help="желаемая высота, см")
    ap.add_argument("--unit", choices=VOCABULARIES, default="plates", help="из чего класть: plates (точнее) или bricks")
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw, если у меша нет своего цвета")
    ap.add_argument("--wheels", nargs="?", const="auto", choices=["auto", *WHEEL_BY_PART],
                    help="найти арки и поставить колёса: auto — подобрать по арке, или номер детали")
    ap.add_argument("--no-tiles", action="store_true", help="не заменять верхние пластины тайлами")
    ap.add_argument("--colors", type=int, default=4, help="до скольких цветов сводить цвет меша")
    ap.add_argument("--symmetric", action="store_true", help="зеркальная кладка, даже если меш кривоват (фото)")
    ap.add_argument("-o", "--out", help="куда писать .ldr")
    args = ap.parse_args()

    vocabulary = VOCABULARIES[args.unit]
    build_at = partial(build, args.mesh, default_color=args.color, vocabulary=vocabulary,
                       wheels=args.wheels, tiles=not args.no_tiles, max_colors=args.colors,
                       force_symmetric=args.symmetric)
    if args.parts:
        result = fit_parts(build_at, args.parts)
    elif args.height:
        result = fit_height(build_at, args.mesh, args.height, vocabulary)
    else:
        result = build_at(args.grid or 30)

    out = Path(args.out) if args.out else Path("out") / f"{Path(args.mesh).stem}_g{result.grid}.ldr"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ldr(result.bricks, str(out), out.stem, result.fixtures)
    print(_summary(result))
    print("->", out)


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
