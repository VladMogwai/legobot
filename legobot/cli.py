"""python -m legobot <mesh> [--grid N] [--color C] [-o out.ldr]"""
import argparse

import numpy as np
from collections import Counter
from pathlib import Path

from .colors import nearest_codes
from .layout import ANY_COLOR, layout_bricks
from .ldraw import write_ldr
from .voxelize import drop_floating, interior, is_symmetric, symmetrize, voxelize_mesh

YELLOW = 14


def main() -> None:
    ap = argparse.ArgumentParser(prog="legobot")
    ap.add_argument("mesh", help="STL/OBJ/GLB, ось Z — вертикаль")
    ap.add_argument("--grid", type=int, default=30, help="штырьков по длинной стороне")
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw, если у меша нет своего цвета")
    ap.add_argument("-o", "--out", help="куда писать .ldr")
    args = ap.parse_args()

    model = voxelize_mesh(args.mesh, args.grid)
    mirrored = is_symmetric(model)
    if mirrored:
        model = symmetrize(model)
    model = drop_floating(model)
    voxels = model.occupancy
    codes = np.full(voxels.shape, ANY_COLOR)
    if model.colors is not None:
        surface = voxels & ~interior(voxels)
        codes[surface] = nearest_codes(model.colors[surface])
        body_color = Counter(codes[surface].tolist()).most_common(1)[0][0]
    else:
        body_color = args.color
    bricks = layout_bricks(voxels, codes, body_color, mirrored=mirrored)

    out = Path(args.out) if args.out else Path("out") / f"{Path(args.mesh).stem}_g{args.grid}.ldr"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ldr(bricks, str(out), out.stem)

    covered = sum(b.part.area for b in bricks)
    assert covered == int(voxels.sum()), f"покрыто {covered} из {int(voxels.sum())}"
    counts = Counter(b.part.label for b in bricks)
    by_color = Counter(b.color for b in bricks)
    print(f"grid {voxels.shape}  voxels {int(voxels.sum())}  layers {len({b.layer for b in bricks})}  "
          f"{'зеркальная кладка' if mirrored else 'модель несимметрична'}")
    print(f"bricks {len(bricks)}: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    print("colors: " + ", ".join(f"{code} x{n}" for code, n in by_color.most_common()))
    print("->", out)


if __name__ == "__main__":
    main()
