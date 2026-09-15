"""Проверка размещения скосов: точки детали после поворота должны лечь ровно в свои клетки,
задняя колонка — полной высоты, передняя — со скосом. Пишет тестовый .ldr для Studio."""
import sys

import numpy as np

from legobot.parts import PLATE_HEIGHT_LDU, STUD_LDU
from legobot.slopes import FACINGS, SLOPES, PlacedSlope, _ROTATION
from tools.ldraw_geometry import points

ok = True
lines = ["0 FILE slopes_test.ldr", "0 slopes_test", "0 Name:  slopes_test", "0 Author:  legobot"]   # заголовок как у write_ldr
row = 0
for slope in SLOPES.values():
    for col, facing in enumerate(FACINGS):
        x, z, top = col * 8, row * 8, 4
        p = PlacedSlope(slope, x, z, top, facing, 14)
        cells = p.cells()
        # ожидаемый бокс из клеток
        xs = [c[0] for c in cells]; zs = [c[1] for c in cells]; ks = [c[2] for c in cells]
        exp_lo = np.array([min(xs) * STUD_LDU, -max(ks) * PLATE_HEIGHT_LDU, min(zs) * STUD_LDU])   # слой k: y от -k*8 (верх) до -(k-1)*8
        exp_hi = np.array([(max(xs) + 1) * STUD_LDU, -(min(ks) - 1) * PLATE_HEIGHT_LDU, (max(zs) + 1) * STUD_LDU])
        t = p.ldraw_line().split()
        pos = np.array(list(map(float, t[2:5]))); M = np.array(list(map(float, t[5:14]))).reshape(3, 3)
        pts = points(slope.number + ".dat") @ M.T + pos
        pts = pts[pts[:, 1] >= exp_lo[1]]   # без штырьков сверху
        lo, hi = pts.min(0), pts.max(0)
        good = np.allclose(lo, exp_lo, atol=1) and np.allclose(hi, exp_hi, atol=1)   # у «сырка» верхнее ребро скруглено
        # самая высокая точка (min y) должна быть над задней колонкой, губка — над передней
        dx, dz = FACINGS[facing]
        front_cell = next((cx, cz) for cx, cz, j in p.columns() if j == slope.depth - 1)
        plane = (dx * (front_cell[0] + 0.5) + dz * (front_cell[1] + 0.5) + 0.5) * STUD_LDU   # передняя грань
        near_front = pts[np.abs(dx * pts[:, 0] + dz * pts[:, 2] - plane) < 1]
        lip = exp_hi[1] - near_front[:, 1].min()
        slope_ok = abs(lip - slope.lip) <= 2
        ok &= good and slope_ok
        print(f"{slope.number:6} {facing}: бокс {'ok' if good else 'НЕ ТОТ ' + str((lo, hi, exp_lo, exp_hi))}, губка {lip:.0f} LDU {'ok' if slope_ok else 'НЕ ТА'}")
        lines.append(p.ldraw_line())
        # опорная пластина под задней колонкой и пластина-ориентир перед губкой
        for cx, cz, j in p.columns():
            lines.append(f"1 4 {(cx + 0.5) * STUD_LDU:.6f} {-(p.layer - 1) * PLATE_HEIGHT_LDU:.6f} {(cz + 0.5) * STUD_LDU:.6f} 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000 3024.dat")
    row += 1
lines += ["0 STEP", "0 NOFILE"]
out = sys.argv[1] if len(sys.argv) > 1 else "out/slopes_test.ldr"
open(out, "w").write("\n".join(lines) + "\n")
print("ВСЁ OK" if ok else "ЕСТЬ ОШИБКИ", "->", out)
