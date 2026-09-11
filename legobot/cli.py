"""python -m legobot <mesh> [--grid N] [--color C] [-o out.ldr]"""
import argparse
from collections import Counter
from pathlib import Path

from .layout import drop_floating, layout_bricks
from .ldraw import write_ldr
from .voxelize import is_symmetric, symmetrize, voxelize_mesh

YELLOW = 14


def main() -> None:
    ap = argparse.ArgumentParser(prog="legobot")
    ap.add_argument("mesh", help="STL/OBJ/GLB, ось Z — вертикаль")
    ap.add_argument("--grid", type=int, default=30, help="штырьков по длинной стороне")
    ap.add_argument("--color", type=int, default=YELLOW, help="код цвета LDraw")
    ap.add_argument("-o", "--out", help="куда писать .ldr")
    args = ap.parse_args()

    voxels = voxelize_mesh(args.mesh, args.grid)
    mirrored = is_symmetric(voxels)
    if mirrored:
        voxels = symmetrize(voxels)
    voxels = drop_floating(voxels)
    bricks = layout_bricks(voxels, mirrored=mirrored)

    out = Path(args.out) if args.out else Path("out") / f"{Path(args.mesh).stem}_g{args.grid}.ldr"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ldr(bricks, str(out), out.stem, args.color)

    covered = sum(b.part.area for b in bricks)
    assert covered == int(voxels.sum()), f"покрыто {covered} из {int(voxels.sum())}"
    counts = Counter(b.part.label for b in bricks)
    print(f"grid {voxels.shape}  voxels {int(voxels.sum())}  layers {len({b.layer for b in bricks})}  "
          f"{'зеркальная кладка' if mirrored else 'модель несимметрична'}")
    print(f"bricks {len(bricks)}: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    print("->", out)


if __name__ == "__main__":
    main()
