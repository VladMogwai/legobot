"""Скосы: детали с наклонной гранью, закрывающие ступеньки на поверхности.

Модель кладётся пластинами (слой 8 LDU), скос занимает 2 или 3 слоя. В родной ориентации
LDraw скос спускается к −Z: сзади (+Z) задняя колонка полной высоты со штырьком (у 45° и 33°),
впереди — колонки под наклонной гранью, заканчивающейся губкой 4 LDU. Скос кладётся «срезая»
угол: все его клетки в исходной модели заняты, поверхность после замены не выше прежней.
"""
from dataclasses import dataclass

import numpy as np

from .parts import PLATE_HEIGHT_LDU, STUD_LDU


@dataclass(frozen=True)
class Slope:
    number: str
    width: int          # поперёк наклона, штырьков
    run: int            # колонок под наклонной гранью
    back: int           # полных колонок сзади (0 у «сырка»)
    layers: int         # высота в пластинах
    origin_top: bool    # начало координат детали: верх (True) или низ (False)
    lip: int            # высота губки спереди, LDU

    @property
    def depth(self) -> int:
        return self.back + self.run

    @property
    def height(self) -> int:
        return self.layers * PLATE_HEIGHT_LDU


SLOPES = {
    "3040b": Slope("3040b", 1, 1, 1, 3, True, 4),    # Slope Brick 45 2 x 1
    "3039": Slope("3039", 2, 1, 1, 3, True, 4),      # Slope Brick 45 2 x 2
    "3037": Slope("3037", 4, 1, 1, 3, True, 4),      # Slope Brick 45 2 x 4
    "54200": Slope("54200", 1, 1, 0, 2, False, 4),   # Slope Brick 31 1 x 1 x 2/3 («сырок»)
    "85984": Slope("85984", 2, 1, 0, 2, False, 4),   # Slope Brick 31 1 x 2 x 2/3
    "4286": Slope("4286", 1, 2, 1, 3, True, 4),      # Slope Brick 33 3 x 1
    "3298": Slope("3298", 2, 2, 1, 3, True, 4),      # Slope Brick 33 3 x 2
}

# Куда спускается грань: смещение на одну клетку в сторону низа.
FACINGS = {"-z": (0, -1), "+z": (0, 1), "-x": (-1, 0), "+x": (1, 0)}
_ACROSS = {"-z": (1, 0), "+z": (-1, 0), "-x": (0, -1), "+x": (0, 1)}


@dataclass(frozen=True)
class PlacedSlope:
    slope: Slope
    x: int          # клетка задней колонки (у «сырка» — самой высокой клетки), левая по ширине
    z: int
    top: int        # верхний слой
    facing: str     # ключ FACINGS
    color: int

    @property
    def layer(self) -> int:
        """Нижний слой — для порядка сборки."""
        return self.top - self.slope.layers + 1

    def columns(self) -> list[tuple[int, int, int]]:
        """(x, z, колонка от задней): 0 — задняя полная, дальше — под наклоном."""
        dx, dz = FACINGS[self.facing]
        ax, az = _across(self.facing)
        out = []
        for j in range(self.slope.depth):
            for i in range(self.slope.width):
                out.append((self.x + dx * j + ax * i, self.z + dz * j + az * i, j))
        return out

    def cells(self) -> list[tuple[int, int, int]]:
        return [(x, z, k) for x, z, _ in self.columns() for k in range(self.layer, self.top + 1)]

    def ldraw_line(self) -> str:
        s = self.slope
        cx, cz = _center(self)
        cy = -self.top * PLATE_HEIGHT_LDU if s.origin_top else -(self.layer - 1) * PLATE_HEIGHT_LDU
        rot = " ".join(f"{v:.6f}" for v in _ROTATION[self.facing].ravel())
        return f"1 {self.color} {cx:.6f} {cy:.6f} {cz:.6f} {rot} {s.number}.dat"


def _across(facing: str) -> tuple[int, int]:
    """Куда после поворота смотрит родная ось +X детали (вдоль ширины)."""
    return _ACROSS[facing]


def _center(p: PlacedSlope) -> tuple[float, float]:
    """Центр детали в LDU: середина задней колонки по ширине (там начало координат детали)."""
    ax, az = _across(p.facing)
    cx = (p.x + 0.5 + ax * (p.slope.width - 1) / 2) * STUD_LDU
    cz = (p.z + 0.5 + az * (p.slope.width - 1) / 2) * STUD_LDU
    return cx, cz


def _rotation_y(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Родная ориентация спускается к −Z; поворот вокруг Y переводит −Z в нужную сторону.
_ROTATION = {"-z": _rotation_y(0), "+z": _rotation_y(180), "-x": _rotation_y(90), "+x": _rotation_y(-90)}
