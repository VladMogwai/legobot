"""Скосы: детали с наклонной гранью, закрывающие ступеньки на поверхности.

Модель кладётся пластинами (слой 8 LDU), скос занимает 2 или 3 слоя. В родной ориентации
LDraw скос спускается к −Z: сзади (+Z) задняя колонка полной высоты со штырьком (у 45° и 33°),
впереди — колонки под наклонной гранью, заканчивающейся губкой 4 LDU. Скос кладётся «срезая»
угол: все его клетки в исходной модели заняты, поверхность после замены не выше прежней.
"""
from dataclasses import dataclass

import numpy as np

from .parts import PLATE_HEIGHT_LDU, STUD_LDU, Part


@dataclass(frozen=True)
class Slope:
    number: str
    width: int          # поперёк наклона, штырьков
    run: int            # колонок под наклонной гранью
    back: int           # полных колонок сзади (0 у «сырка»)
    layers: int         # высота в пластинах
    origin_top: bool    # начало координат детали: верх (True) или низ (False)
    lip: int            # высота губки спереди, LDU
    front_drop: int = 0 # на сколько слоёв передняя колонка ниже верха (у 33° грань над ней ниже t−1)

    @property
    def depth(self) -> int:
        return self.back + self.run

    @property
    def height(self) -> int:
        return self.layers * PLATE_HEIGHT_LDU

    def profile(self) -> list[tuple[float, float]]:
        """Высота поверхности (в слоях от низа детали) у заднего и переднего края каждой колонки."""
        lip = self.lip / PLATE_HEIGHT_LDU
        step = (self.layers - lip) / self.run
        return [(self.layers, self.layers)] * self.back + [
            (self.layers - step * r, self.layers - step * (r + 1)) for r in range(self.run)]


SLOPES = {
    "3040b": Slope("3040b", 1, 1, 1, 3, True, 4),    # Slope Brick 45 2 x 1
    "3039": Slope("3039", 2, 1, 1, 3, True, 4),      # Slope Brick 45 2 x 2
    "3037": Slope("3037", 4, 1, 1, 3, True, 4),      # Slope Brick 45 2 x 4
    "54200": Slope("54200", 1, 1, 0, 2, False, 4),   # Slope Brick 31 1 x 1 x 2/3 («сырок»)
    "85984": Slope("85984", 2, 1, 0, 2, False, 4),   # Slope Brick 31 1 x 2 x 2/3
    "4286": Slope("4286", 1, 2, 1, 3, True, 4, 1),   # Slope Brick 33 3 x 1
    "3298": Slope("3298", 2, 2, 1, 3, True, 4, 1),   # Slope Brick 33 3 x 2
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
        """Клетки, которые деталь занимает физически."""
        s = self.slope
        return [(x, z, k) for x, z, j in self.columns()
                for k in range(self.layer, self.top + 1 - (s.front_drop if j == s.depth - 1 else 0))]

    def ldraw_line(self) -> str:
        s = self.slope
        cx, cz = _center(self)
        cy = -self.top * PLATE_HEIGHT_LDU if s.origin_top else -(self.layer - 1) * PLATE_HEIGHT_LDU
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
# Условия на колонки перед постановкой (в слоях относительно верха t; d — направление вниз):
#   45°: задняя A' и передняя A заняты t-2..t, над A пусто, перед A (B) пусто t-2..t.
#   33°: A'' и A' заняты t-2..t, над A' пусто, A занята t-2..t-1 и пуста на t, B пуста t-2..t.
#   «сырок»: A занята t-1..t, над A пусто, A' занята на t, B пуста t-1..t.
_KINDS = [  # (номера по ширине, проверка колонок от передней к задней)
    ({1: "4286", 2: "3298"}, "33"),
    ({1: "3040b", 2: "3039", 4: "3037"}, "45"),
    ({1: "54200", 2: "85984"}, "cheese"),
]


MAX_NORMAL_DEVIATION = 30  # градусов между нормалью меша и гранью скоса


def find_slopes(occupancy: np.ndarray, codes: np.ndarray, default_color: int, mirrored: bool,
                any_color: int, normals: np.ndarray | None = None) -> tuple[list[PlacedSlope], np.ndarray]:
    """Скосы на ступеньках модели, крупные первыми. Возвращает (скосы, номер скоса на клетку или -1).
    При зеркальной кладке скос ставится вместе с отражением или не ставится вовсе.
    normals — нормали меша по вокселям: скос ставится, только если грань смотрит туда же, куда
    поверхность (иначе на круглом теле скосы встают вразнобой)."""
    nx, nz, nk = occupancy.shape
    taken = np.full(occupancy.shape, -1)

    def filled(x, z, k):
        return 0 <= x < nx and 0 <= z < nz and 0 <= k < nk and occupancy[x, z, k]

    def free(x, z, k):
        return 0 <= x < nx and 0 <= z < nz and 0 <= k < nk and taken[x, z, k] < 0

    def matches(kind, x, z, t, dx, dz):
        bx, bz = x + dx, z + dz
        ax, az = x - dx, z - dz
        if kind == "45":
            return (all(filled(x, z, k) and filled(ax, az, k) for k in range(t - 2, t + 1))
                    and not filled(x, z, t + 1) and not any(filled(bx, bz, k) for k in range(t - 2, t + 1)))
        if kind == "33":
            aax, aaz = x - 2 * dx, z - 2 * dz
            return (all(filled(ax, az, k) and filled(aax, aaz, k) for k in range(t - 2, t + 1))
                    and not filled(ax, az, t + 1) and filled(x, z, t - 2) and filled(x, z, t - 1)
                    and not filled(x, z, t) and not any(filled(bx, bz, k) for k in range(t - 2, t + 1)))
        return (filled(x, z, t - 1) and filled(x, z, t) and not filled(x, z, t + 1) and filled(ax, az, t)
                and not filled(bx, bz, t - 1) and not filled(bx, bz, t))

    placed: list[PlacedSlope] = []

    def place(p: PlacedSlope) -> None:
        for c in p.cells():
            taken[c] = len(placed)
        placed.append(p)

    for by_width, kind in _KINDS:
        for width in sorted(by_width, reverse=True):
            slope = SLOPES[by_width[width]]
            for facing, (dx, dz) in FACINGS.items():
                ax, az = _ACROSS[facing]
                for t in range(nk - 1, slope.layers - 2, -1):
                    for x in range(nx):
                        for z in range(nz):
                            if not all(matches(kind, x + ax * i, z + az * i, t, dx, dz) for i in range(width)):
                                continue
                            back = (x - dx * (slope.depth - 1), z - dz * (slope.depth - 1))
                            p = PlacedSlope(slope, *back, t, facing, default_color)
                            color = _slope_color(p, codes, default_color, any_color)
                            if color is None:
                                continue   # разноцветные клетки (глаз, край клюва) — скос их смешал бы
                            if normals is not None and not _faces_surface(p, normals):
                                continue
                            p = PlacedSlope(slope, *back, t, facing, color)
                            pair = [p] if not mirrored else [p, mirrored_slope(p, nx)]
                            if pair[-1].cells() == p.cells():
                                pair = [p]
                            cells = [c for q in pair for c in q.cells()]
                            if len(set(cells)) == len(cells) and all(free(*c) for c in cells):
                                for q in pair:
                                    place(q)
    return placed, taken


def _faces_surface(p: PlacedSlope, normals) -> bool:
    """Средняя нормаль поверхности в клетках скоса близка к нормали его грани."""
    n = sum((normals[c] for c in p.cells()), np.zeros(3))
    if np.linalg.norm(n) < 1e-9:
        return False
    dx, dz = FACINGS[p.facing]
    rise = (p.slope.layers - p.slope.lip / PLATE_HEIGHT_LDU) * PLATE_HEIGHT_LDU / STUD_LDU / p.slope.run   # штырьков вниз на штырёк вперёд
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
    return PlacedSlope(p.slope, nx - 1 - last[0], last[1], p.top, facing, p.color)
