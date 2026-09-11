"""Фиксированные детали: ставятся в заданное место как есть, укладка их обходит.

Пока одна сборка — колесо: Technic-кирпич 1x2 с осевым отверстием внутри кузова,
ось через него, колесо на оси снаружи. Арки находятся по бортам модели.
"""
from dataclasses import dataclass

import numpy as np

from .parts import STUD_LDU
from .voxelize import VoxelModel

IDENTITY = (1, 0, 0, 0, 1, 0, 0, 0, 1)
ROTATE_Y90 = (0, 0, 1, 0, 1, 0, -1, 0, 0)  # локальная ось Z детали -> мировая X

DARK_GRAY = 72
BLACK = 0


@dataclass(frozen=True)
class Fixture:
    part: str
    color: int
    position: tuple[float, float, float]   # LDU, ось Y вниз
    rotation: tuple[float, ...] = IDENTITY


@dataclass(frozen=True)
class WheelSpec:
    part: str        # сборка «диск + шина», ось вдоль локальной Z
    diameter: float  # LDU
    width: float     # LDU


WHEELS = {
    "city": WheelSpec("56904c01", 107.1, 38.0),      # обычная машина, ~5.4 штырька
    "technic": WheelSpec("3739c01", 205.2, 60.0),    # крупная, ~10 штырьков
}

AXLE_BRICK = "32064a"   # Technic Brick 1 x 2 with Axlehole; отверстие на 10 LDU ниже верха
AXLE = "4519"           # Technic Axle 3, длина 60 вдоль X
AXLE_LENGTH = 60
AXLE_BRICK_HEIGHT = 24


@dataclass
class Arch:
    side: int          # -1 левый борт (малый X), +1 правый
    x_face: float      # X внешней грани борта, LDU
    z_center: float    # LDU
    y_center: float    # LDU (вниз)
    radius: float      # LDU


def wheel_fixtures(model: VoxelModel, layer_ldu: int, wheel: WheelSpec) -> tuple[list[Fixture], np.ndarray]:
    """Возвращает детали колёс и маску вокселей, которые надо освободить под них."""
    fixtures, cleared = [], np.zeros(model.shape, dtype=bool)
    for arch in _find_arches(model.occupancy, layer_ldu):
        fixtures.extend(_wheel_assembly(arch, wheel))
        cleared |= _clearance(model.shape, layer_ldu, arch, wheel)
    return fixtures, cleared


def _find_arches(occ: np.ndarray, layer_ldu: int) -> list[Arch]:
    """Арка — участок борта, где нижний край кузова приподнят над порогом."""
    nx, nz, nl = occ.shape
    arches = []
    for side, columns in ((-1, range(nx)), (+1, range(nx - 1, -1, -1))):
        outer = next(x for x in columns if occ[x].any())
        wall = occ[outer:outer + 2] if side < 0 else occ[outer - 1:outer + 1]
        wall = wall.any(axis=0)                       # [z, layer]
        floor = np.where(wall.any(1), wall.argmax(1), -1)
        present = floor >= 0
        baseline = np.bincount(floor[present]).argmax()
        raised = present & (floor > baseline + 1)
        x_face = outer * STUD_LDU if side < 0 else (outer + 1) * STUD_LDU
        for z0, z1 in _runs(raised):
            if z1 - z0 < 3:
                continue
            top_layer = floor[z0:z1].max()
            radius = (z1 - z0) / 2 * STUD_LDU
            arches.append(Arch(
                side=side, x_face=x_face,
                z_center=(z0 + z1) / 2 * STUD_LDU,
                y_center=-top_layer * layer_ldu + layer_ldu + radius,  # верх арки — низ слоя top_layer
                radius=radius,
            ))
    return arches


def _runs(mask: np.ndarray):
    runs, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    return runs


def _wheel_assembly(arch: Arch, wheel: WheelSpec) -> list[Fixture]:
    inward = -arch.side                                   # направление внутрь кузова по X
    # Кирпич с осью сидит на границах слоёв; ось колеса — по отверстию кирпича.
    brick_top = 8 * round((arch.y_center - 10) / 8)
    axle_y = brick_top + 10
    wheel_x = arch.x_face + inward * wheel.width / 2      # внешняя грань колеса вровень с бортом
    brick_x = wheel_x + inward * (wheel.width / 2 + STUD_LDU / 2)
    brick_z = STUD_LDU * round(arch.z_center / STUD_LDU)  # кирпич 1x2 симметричен относительно оси колеса
    axle_x = wheel_x + inward * (wheel.width / 2 + STUD_LDU - AXLE_LENGTH / 2)
    return [
        Fixture(AXLE_BRICK, DARK_GRAY, (brick_x, brick_top, brick_z), ROTATE_Y90),
        Fixture(AXLE, DARK_GRAY, (axle_x, axle_y, arch.z_center), IDENTITY),
        Fixture(wheel.part, BLACK, (wheel_x, axle_y, arch.z_center), ROTATE_Y90),
    ]


def _clearance(shape, layer_ldu, arch: Arch, wheel: WheelSpec) -> np.ndarray:
    """Воксели под колесо (цилиндр с зазором) и под кирпич с осью."""
    nx, nz, nl = shape
    inward = -arch.side
    xs = (np.arange(nx) + 0.5) * STUD_LDU
    zs = (np.arange(nz) + 0.5) * STUD_LDU
    ys = -np.arange(nl) * layer_ldu + layer_ldu / 2   # в LDraw Y вниз: слой k занимает [-k*h, -k*h + h]
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")

    wheel_x = arch.x_face + inward * wheel.width / 2
    brick_top = 8 * round((arch.y_center - 10) / 8)
    axle_y = brick_top + 10
    in_width = np.abs(X - wheel_x) <= wheel.width / 2 + STUD_LDU / 2
    in_disc = np.hypot(Z - arch.z_center, Y - axle_y) <= wheel.diameter / 2 + STUD_LDU / 2
    cleared = in_width & in_disc

    brick_x = wheel_x + inward * (wheel.width / 2 + STUD_LDU / 2)
    brick_z = STUD_LDU * round(arch.z_center / STUD_LDU)
    in_brick = (
        (np.abs(X - brick_x) <= STUD_LDU / 2)
        & (np.abs(Z - brick_z) <= STUD_LDU)
        & (Y >= brick_top) & (Y <= brick_top + AXLE_BRICK_HEIGHT)
    )
    return cleared | in_brick
