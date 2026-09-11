"""Сборка целиком: меш -> воксели -> кирпичи. Одна функция для CLI, подбора размера и бота."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .colors import nearest_codes
from .layout import ANY_COLOR, PlacedBrick, layout_bricks
from .parts import BRICK_HEIGHT_LDU, STUD_LDU
from .voxelize import VoxelModel, drop_floating, interior, symmetrize, voxelize_mesh

STUD_MM = 8.0
BRICK_MM = STUD_MM * BRICK_HEIGHT_LDU / STUD_LDU  # 9.6


@dataclass
class Build:
    bricks: list[PlacedBrick]
    model: VoxelModel
    grid: int
    mirrored: bool

    @property
    def part_count(self) -> int:
        return len(self.bricks)

    @property
    def size_mm(self) -> tuple[float, float, float]:
        """Габариты в миллиметрах: ширина (X), длина (Z), высота."""
        occ = self.model.occupancy
        x, z, k = (np.ptp(np.nonzero(occ)[i]) + 1 for i in range(3))
        return x * STUD_MM, z * STUD_MM, k * BRICK_MM

    @property
    def colors(self) -> Counter:
        return Counter(b.color for b in self.bricks)


def build(mesh_path: str, grid: int, default_color: int) -> Build:
    model = voxelize_mesh(mesh_path, grid)
    mirrored = model.symmetric
    if mirrored:
        model = symmetrize(model)
    model = drop_floating(model)

    voxels = model.occupancy
    codes = np.full(voxels.shape, ANY_COLOR)
    body_color = default_color
    if model.colors is not None:
        surface = voxels & ~interior(voxels)
        codes[surface] = nearest_codes(model.colors[surface])
        body_color = Counter(codes[surface].tolist()).most_common(1)[0][0]

    bricks = layout_bricks(voxels, codes, body_color, mirrored=mirrored)
    covered = sum(b.part.area for b in bricks)
    assert covered == int(voxels.sum()), f"покрыто {covered} из {int(voxels.sum())} вокселей"
    return Build(bricks, model, grid, mirrored)
