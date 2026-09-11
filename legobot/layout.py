"""Воксельная сетка -> раскладка кирпичей по слоям.

Жадная укладка «трудные клетки первыми». В каждом слое клетки сортируются
по открытости — сколько вокруг пусто: сбоку, снизу, сверху. Краевые клетки
раскладываются первыми, пока внутренность свободна, и кирпич может лечь
от края внутрь, на полную опору. Для клетки перебираются все кирпичи
во всех положениях, которые её накрывают; выбирается тот, что
  1) сшивает ещё не связанные куски модели,
  2) держится большим числом штырьков снизу и сверху,
  3) крупнее.
Слои чередуют ведущую ось — при равном счёте кирпич ложится поперёк
кирпичей соседнего слоя.
"""
from dataclasses import dataclass

import numpy as np

from .parts import Part, Vocabulary

MIN_SUPPORT = 0.5   # доля площади, которая должна лежать на соседнем слое
NO_BRICK = -1
GROUND = 0          # id «земли» — сплошной опоры под нулевым слоем
ANY_COLOR = -1      # воксель без требования к цвету (внутренний, снаружи не виден)


@dataclass(frozen=True)
class PlacedBrick:
    part: Part
    x: int         # левая клетка по X
    z: int         # ближняя клетка по Z
    layer: int     # 0 = нижний
    rotated: bool  # повёрнут на 90° относительно родной ориентации
    color: int     # код цвета LDraw

    @property
    def width(self) -> int:
        return self.part.length if self.rotated else self.part.width

    @property
    def length(self) -> int:
        return self.part.width if self.rotated else self.part.length


class _Components:
    """Union-find: какие кирпичи уже связаны друг с другом через штырьки."""

    def __init__(self):
        self._parent = []

    def add(self) -> int:
        self._parent.append(len(self._parent))
        return len(self._parent) - 1

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self._parent[self.find(i)] = self.find(j)


def layout_bricks(voxels: np.ndarray, colors: np.ndarray, default_color: int, vocabulary: Vocabulary,
                  mirrored: bool = False) -> list[PlacedBrick]:
    """voxels: bool [x, y, z], z — вертикаль. colors: код цвета LDraw на каждый воксель той же формы,
    ANY_COLOR — воксель без требования. Кирпич одноцветный: все его воксели с требованием одного цвета;
    если требований нет — default_color. mirrored — зеркальная кладка относительно середины X."""
    nx, nz, nlayers = voxels.shape
    footprints = {(p.width, p.length) for p in vocabulary.parts} | {(p.length, p.width) for p in vocabulary.parts}
    placed: list[PlacedBrick] = []
    components = _Components()
    components.add()  # GROUND
    below_ids = np.full((nx, nz), GROUND)
    empty = np.zeros((nx, nz), dtype=bool)
    for k in range(nlayers):
        layer = voxels[:, :, k]
        if not layer.any():
            below_ids = np.full((nx, nz), NO_BRICK)
            continue
        above = voxels[:, :, k + 1] if k + 1 < nlayers else empty
        below_ids = _layout_layer(layer, colors[:, :, k], default_color, below_ids, above, k,
                                  placed, components, mirrored, vocabulary, footprints)
    return placed


def _layout_layer(layer, layer_colors, default_color, below_ids, above, k, placed, components, mirrored,
                  vocabulary, footprints):
    free = layer.copy()
    nx, nz = free.shape
    ids = np.full((nx, nz), NO_BRICK)
    below = below_ids != NO_BRICK
    along_x = k % 2 == 1

    def place(x0, z0, w, l):
        brick_id = components.add()
        for under_id in np.unique(below_ids[x0:x0 + w, z0:z0 + l]):
            if under_id != NO_BRICK:
                components.union(brick_id, int(under_id))
        free[x0:x0 + w, z0:z0 + l] = False
        ids[x0:x0 + w, z0:z0 + l] = brick_id
        placed.append(_brick(vocabulary, w, l, x0, z0, k, _brick_color(layer_colors[x0:x0 + w, z0:z0 + l], default_color)))

    for x, z in _cells_hardest_first(layer, below, above):
        if not free[x, z] or (mirrored and x > (nx - 1) // 2):
            continue  # при зеркальной кладке правую половину заполняют отражения
        x0, z0, w, l = _best_placement(free, layer_colors, below_ids, below, above, x, z, along_x, components, mirrored, footprints)
        place(x0, z0, w, l)
        mx0 = nx - x0 - w
        if mirrored and mx0 != x0:
            place(mx0, z0, w, l)
    return ids


def _cells_hardest_first(layer, below, above):
    """Клетки слоя, самые открытые первыми: у них меньше всего вариантов."""
    padded = np.pad(layer, 1)
    side_empty = sum(
        ~padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dz:padded.shape[1] - 1 + dz]
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )
    exposure = side_empty + (~below) + (~above)
    xs, zs = np.nonzero(layer)
    order = np.argsort(-exposure[xs, zs], kind="stable")
    return list(zip(xs[order].tolist(), zs[order].tolist()))


def _best_placement(free, colors, below_ids, below, above, cx, cz, along_x, components, mirrored, footprints):
    nx, nz = free.shape
    best, best_key = None, None
    for w, l in footprints:
        for ox in range(w):
            for oz in range(l):
                x0, z0 = cx - ox, cz - oz
                x1, z1 = x0 + w, z0 + l
                if x0 < 0 or z0 < 0 or x1 > nx or z1 > nz or not free[x0:x1, z0:z1].all():
                    continue
                if not _single_color(colors[x0:x1, z0:z1]):
                    continue
                if mirrored and not _mirror_fits(free, colors, x0, x1, z0, z1):
                    continue
                area = w * l
                studs_below = int(below[x0:x1, z0:z1].sum())
                studs_above = int(above[x0:x1, z0:z1].sum())
                supported = max(studs_below, studs_above) / area >= MIN_SUPPORT
                under = below_ids[x0:x1, z0:z1]
                pieces = {components.find(int(i)) for i in np.unique(under) if i != NO_BRICK}
                key = (
                    len(pieces) >= 2,            # сшивает куски
                    supported,                   # стоит хотя бы наполовину
                    studs_below + studs_above,   # чем держится
                    area,                        # крупнее
                    (w > l) == along_x,          # поперёк соседнего слоя
                )
                if best_key is None or key > best_key:
                    best, best_key = (x0, z0, w, l), key
    return best


def _mirror_fits(free, colors, x0, x1, z0, z1):
    """Кирпич либо сам симметричен относительно середины X, либо его отражение
    свободно, того же цвета и не задевает его."""
    nx = free.shape[0]
    mx0, mx1 = nx - x1, nx - x0
    if (mx0, mx1) == (x0, x1):
        return True
    return (mx1 <= x0 or mx0 >= x1) and free[mx0:mx1, z0:z1].all() and _single_color(colors[mx0:mx1, z0:z1])


def _single_color(region) -> bool:
    """Все воксели с требованием к цвету — одного цвета."""
    required = region[region != ANY_COLOR]
    return required.size == 0 or (required == required[0]).all()


def _brick_color(region, default_color) -> int:
    required = region[region != ANY_COLOR]
    return int(required[0]) if required.size else default_color


def _brick(vocabulary, w, l, x, z, k, color):
    part = vocabulary.by_size(w, l)
    if part is not None:
        return PlacedBrick(part, x, z, k, rotated=False, color=color)
    return PlacedBrick(vocabulary.by_size(l, w), x, z, k, rotated=True, color=color)
