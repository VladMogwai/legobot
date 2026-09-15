"""Скосы: детали с наклонной гранью, закрывающие ступеньки на поверхности.

Слой кладки — пластина (8 LDU) или кирпич (24): скос занимает целое число слоёв. В родной ориентации
LDraw скос спускается к −Z: сзади (+Z) задняя колонка полной высоты со штырьком (у 45° и 33°),
впереди — колонки под наклонной гранью, заканчивающейся губкой 4 LDU. Скос кладётся «срезая»
угол: все его клетки в исходной модели заняты, поверхность после замены не выше прежней.
"""
from dataclasses import dataclass

import numpy as np

from .parts import STUD_LDU, Part


@dataclass(frozen=True)
class Slope:
    number: str
    width: int          # поперёк наклона, штырьков
    run: int            # колонок под наклонной гранью
    back: int           # полных колонок сзади (0 у «сырка»)
    height: int         # LDU
    origin_top: bool    # начало координат детали: верх (True) или низ (False)
    lip: int            # высота губки спереди, LDU

    @property
    def depth(self) -> int:
        return self.back + self.run

    def layers(self, layer_ldu: int) -> int | None:
        """Сколько слоёв кладки занимает; None, если не делится (сырок в кирпичной кладке)."""
        return self.height // layer_ldu if self.height % layer_ldu == 0 else None

    def profile(self, layer_ldu: int) -> list[tuple[float, float]]:
        """Высота поверхности (в слоях от низа детали) у заднего и переднего края каждой колонки."""
        top, lip = self.height / layer_ldu, self.lip / layer_ldu
        step = (top - lip) / self.run
        return [(top, top)] * self.back + [(top - step * r, top - step * (r + 1)) for r in range(self.run)]


SLOPES = {
    "3040b": Slope("3040b", 1, 1, 1, 24, True, 4),   # Slope Brick 45 2 x 1
    "3039": Slope("3039", 2, 1, 1, 24, True, 4),     # Slope Brick 45 2 x 2
    "3037": Slope("3037", 4, 1, 1, 24, True, 4),     # Slope Brick 45 2 x 4
    "54200": Slope("54200", 1, 1, 0, 16, False, 4),  # Slope Brick 31 1 x 1 x 2/3 («сырок»)
    "85984": Slope("85984", 2, 1, 0, 16, False, 4),  # Slope Brick 31 1 x 2 x 2/3
    "4286": Slope("4286", 1, 2, 1, 24, True, 4),     # Slope Brick 33 3 x 1
    "3298": Slope("3298", 2, 2, 1, 24, True, 4),     # Slope Brick 33 3 x 2
}

# Куда спускается грань: смещение на одну клетку в сторону низа.
FACINGS = {"-z": (0, -1), "+z": (0, 1), "-x": (-1, 0), "+x": (1, 0)}
_ACROSS = {"-z": (1, 0), "+z": (-1, 0), "-x": (0, -1), "+x": (0, 1)}


@dataclass(frozen=True)
class PlacedSlope:
    slope: Slope
    x: int          # клетка задней колонки (у «сырка» — самой высокой клетки), первая по ширине
    z: int
    top: int        # верхний слой
    facing: str     # ключ FACINGS
    color: int
    layer_ldu: int  # высота слоя кладки (пластина 8 или кирпич 24)

    @property
    def layers(self) -> int:
        return self.slope.layers(self.layer_ldu)

    @property
    def layer(self) -> int:
        """Нижний слой — для порядка сборки."""
        return self.top - self.layers + 1

    @property
    def width(self) -> int:
        """Занимаемая площадь по X (как у PlacedBrick)."""
        return self.slope.width if self.facing in ("-z", "+z") else self.slope.depth

    @property
    def length(self) -> int:
        return self.slope.depth if self.facing in ("-z", "+z") else self.slope.width

    @property
    def part(self) -> Part:
        return Part(self.slope.number, self.width, self.length, self.slope.height)

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
        """Клетки, которые деталь занимает (у наклонных колонок — до верха детали, с запасом)."""
        return [(x, z, k) for x, z, _ in self.columns() for k in range(self.layer, self.top + 1)]

    def ldraw_line(self) -> str:
        s = self.slope
        cx, cz = _center(self)
        cy = -self.top * self.layer_ldu if s.origin_top else -(self.layer - 1) * self.layer_ldu
        rot = " ".join(f"{v + 0.0:.6f}" for v in _ROTATION[self.facing].ravel())   # без «-0.000000»
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


# --- Поиск ступенек ---
# Скос подходит, если его профиль совпадает с объёмом меша: у задних колонок все слои заняты
# (они несущие), у наклонных — заполнение колонки (сумма долей по слоям) близко к высоте
# грани, а перед губкой и над гранью пусто. Так скос сам решает, «срезать» угол ступеньки
# или «дорисовать» его в полупустую клетку.
_KINDS = [
    {1: "4286", 2: "3298"},            # 33°
    {1: "3040b", 2: "3039", 4: "3037"},  # 45°
    {1: "54200", 2: "85984"},          # «сырок»
]
MAX_FIT_ERROR = 0.35       # слоёв: среднее расхождение объёма колонки с профилем
MAX_NORMAL_DEVIATION = 30  # градусов между нормалью меша и гранью скоса


def find_slopes(occupancy: np.ndarray, codes: np.ndarray, default_color: int, mirrored: bool,
                any_color: int, layer_ldu: int, normals: np.ndarray | None = None,
                fraction: np.ndarray | None = None) -> tuple[list[PlacedSlope], np.ndarray]:
    """Скосы на ступеньках модели: все подходящие места, лучшие (крупнее, точнее) первыми.
    Возвращает (скосы, номер скоса на клетку или -1). При зеркальной кладке скос ставится
    вместе с отражением или не ставится вовсе."""
    nx, nz, nk = occupancy.shape
    if fraction is None:
        fraction = occupancy.astype(np.float32)
    candidates = []
    for by_width in _KINDS:
        for width, number in by_width.items():
            slope = SLOPES[number]
            L = slope.layers(layer_ldu)
            if L is None:
                continue
            for facing in FACINGS:
                for t in range(L - 1, nk):
                    for x in range(nx):
                        for z in range(nz):
                            p = PlacedSlope(slope, x, z, t, facing, default_color, layer_ldu)
                            error = _fit_error(p, occupancy, fraction)
                            if error is None or error > MAX_FIT_ERROR:
                                continue
                            color = _slope_color(p, codes, default_color, any_color)
                            if color is None:
                                continue   # разноцветные клетки (глаз, край клюва) — скос их смешал бы
                            if normals is not None and not _faces_surface(p, normals):
                                continue
                            candidates.append((-(width * slope.depth), error, PlacedSlope(slope, x, z, t, facing, color, layer_ldu)))
    candidates.sort(key=lambda c: c[:2])

    taken = np.full(occupancy.shape, -1)
    placed: list[PlacedSlope] = []
    for _, _, p in candidates:
        pair = [p] if not mirrored else [p, mirrored_slope(p, nx)]
        if pair[-1].cells() == p.cells():
            pair = [p]
        cells = [c for q in pair for c in q.cells()]
        if len(set(cells)) == len(cells) and all(taken[c] < 0 for c in cells):
            for q in pair:
                for c in q.cells():
                    taken[c] = len(placed)
                placed.append(q)
    return placed, taken


def _fit_error(p: PlacedSlope, occupancy, fraction) -> float | None:
    """Среднее расхождение (в слоях) объёма наклонных колонок с профилем; None — место не подходит."""
    nx, nz, nk = occupancy.shape
    dx, dz = FACINGS[p.facing]
    span = range(p.layer, p.top + 1)
    profile = p.slope.profile(p.layer_ldu)
    if p.layer < 0:
        return None

    def filled(x, z, k):
        return 0 <= x < nx and 0 <= z < nz and 0 <= k < nk and occupancy[x, z, k]

    errors = []
    for x, z, j in p.columns():
        if not (0 <= x < nx and 0 <= z < nz):
            return None
        if j < p.slope.back:
            if not all(filled(x, z, k) for k in span):
                return None
            continue
        if filled(x, z, p.top + 1):
            return None
        if j == p.slope.depth - 1 and any(filled(x + dx, z + dz, k) for k in span):
            return None   # перед губкой должно быть пусто
        hb, hf = profile[j]
        errors.append(abs(float(fraction[x, z, p.layer:p.top + 1].sum()) - (hb + hf) / 2))
    return sum(errors) / len(errors)


def _faces_surface(p: PlacedSlope, normals) -> bool:
    """Средняя нормаль поверхности под гранью скоса близка к нормали грани. Берём наклонные
    колонки и клетку перед ними; если поверхности там нет — все клетки детали."""
    dx, dz = FACINGS[p.facing]
    run = [(x, z, k) for x, z, j in p.columns() if j >= p.slope.back for k in range(p.layer - 1, p.top + 1)]
    run += [(x + dx, z + dz, k) for x, z, j in p.columns() if j == p.slope.depth - 1 for k in range(p.layer - 1, p.top + 1)]
    inside = lambda c: all(0 <= c[i] < normals.shape[i] for i in range(3))
    n = sum((normals[c] for c in run if inside(c)), np.zeros(3))
    if np.linalg.norm(n) < 1e-9:
        n = sum((normals[c] for c in p.cells() if inside(c)), np.zeros(3))
    if np.linalg.norm(n) < 1e-9:
        return False
    rise = (p.slope.height - p.slope.lip) / STUD_LDU / p.slope.run   # штырьков вниз на штырёк вперёд
    face = np.array([dx * rise, dz * rise, 1.0])
    cos = n @ face / np.linalg.norm(n) / np.linalg.norm(face)
    return cos >= np.cos(np.radians(MAX_NORMAL_DEVIATION))


def _slope_color(p: PlacedSlope, codes, default_color, any_color) -> int | None:
    """Цвет клеток скоса; None, если клетки требуют разных цветов."""
    colors = {int(codes[c]) for c in p.cells() if codes[c] != any_color}
    if len(colors) > 1:
        return None
    return colors.pop() if colors else default_color


def mirrored_slope(p: PlacedSlope, nx: int) -> PlacedSlope:
    """Отражение относительно середины X: первая клетка по ширине становится последней."""
    facing = {"-x": "+x", "+x": "-x"}.get(p.facing, p.facing)
    ax, az = _ACROSS[p.facing]
    last = (p.x + ax * (p.slope.width - 1), p.z + az * (p.slope.width - 1))
    return PlacedSlope(p.slope, nx - 1 - last[0], last[1], p.top, facing, p.color, p.layer_ldu)
