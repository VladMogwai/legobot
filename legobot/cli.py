"""python -m legobot <mesh> (--grid N | --parts N | --height CM) [--color C] [-o out.ldr]"""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path

from .colors import load_palette
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
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw, если у меша нет своего цвета")
    ap.add_argument("-o", "--out", help="куда писать .ldr")
    args = ap.parse_args()

    build_at = partial(build, args.mesh, default_color=args.color)
    if args.parts:
        result = fit_parts(build_at, args.parts)
    elif args.height:
        result = fit_height(build_at, args.mesh, args.height)
    else:
        result = build_at(args.grid or 30)

    out = Path(args.out) if args.out else Path("out") / f"{Path(args.mesh).stem}_g{result.grid}.ldr"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ldr(result.bricks, str(out), out.stem)
    print(_summary(result))
    print("->", out)


def _summary(r: Build) -> str:
    names = {c.code: c.name for c in load_palette()}
    w, l, h = r.size_mm
    parts = Counter(b.part.label for b in r.bricks)
    return (
        f"сетка {r.grid}, {'зеркальная кладка' if r.mirrored else 'модель несимметрична'}\n"
        f"размер {w / 10:.1f} x {l / 10:.1f} x {h / 10:.1f} см, слоёв {len({b.layer for b in r.bricks})}\n"
        f"деталей {r.part_count}: " + ", ".join(f"{k} x{n}" for k, n in sorted(parts.items())) + "\n"
        f"цвета: " + ", ".join(f"{names.get(c, c)} x{n}" for c, n in r.colors.most_common())
    )


if __name__ == "__main__":
    main()
