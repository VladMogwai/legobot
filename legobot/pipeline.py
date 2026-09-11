"""Сборка целиком: меш -> воксели -> кирпичи. Одна функция для CLI, подбора размера и бота."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .colors import nearest_codes
from .finish import tile_exposed_tops
from .fixtures import WHEEL_BY_PART, Fixture, wheel_fixtures
from .layout import ANY_COLOR, EXPOSED_TOP, PlacedBrick, layout_bricks
from .parts import STUD_LDU, Vocabulary
from .voxelize import VoxelModel, drop_floating, interior, symmetrize, voxelize_mesh

STUD_MM = 8.0


@dataclass
class Build:
    bricks: list[PlacedBrick]
    model: VoxelModel
    grid: int
    mirrored: bool
    vocabulary: Vocabulary
    fixtures: list[Fixture]

    @property
    def layer_mm(self) -> float:
        return STUD_MM * self.vocabulary.height / STUD_LDU

    @property
    def part_count(self) -> int:
        return len(self.bricks) + len(self.fixtures)

    @property
    def size_mm(self) -> tuple[float, float, float]:
        """Габариты в миллиметрах: ширина (X), длина (Z), высота."""
        occ = self.model.occupancy
        x, z, k = (np.ptp(np.nonzero(occ)[i]) + 1 for i in range(3))
        return x * STUD_MM, z * STUD_MM, k * self.layer_mm

    @property
    def colors(self) -> Counter:
        return Counter(b.color for b in self.bricks)


def build(mesh_path: str, grid: int, default_color: int, vocabulary: Vocabulary,
          wheels: str | None = None, tiles: bool = True) -> Build:
    """wheels: None — без колёс, "auto" — подобрать по арке, иначе номер детали колеса."""
    model = voxelize_mesh(mesh_path, grid, vocabulary.aspect)
    mirrored = model.symmetric
    if mirrored:
        model = symmetrize(model)
    fixtures: list[Fixture] = []
    if wheels:
        spec = None if wheels == "auto" else WHEEL_BY_PART[wheels]
        fixtures, cleared = wheel_fixtures(model, vocabulary.height, spec)
        model.occupancy &= ~cleared
    model = drop_floating(model)

    voxels = model.occupancy
    codes = np.full(voxels.shape, ANY_COLOR)
    body_color = default_color
    if model.colors is not None:
        surface = voxels & ~interior(voxels)
        codes[surface] = nearest_codes(model.colors[surface])
        body_color = Counter(codes[surface].tolist()).most_common(1)[0][0]

    if tiles:
        exposed = voxels.copy()
        exposed[:, :, :-1] &= ~voxels[:, :, 1:]
        codes[exposed] = np.where(codes[exposed] == ANY_COLOR, body_color, codes[exposed]) | EXPOSED_TOP
    bricks = layout_bricks(voxels, codes, body_color, vocabulary, mirrored=mirrored)
    covered = sum(b.part.area for b in bricks)
    assert covered == int(voxels.sum()), f"покрыто {covered} из {int(voxels.sum())} вокселей"
    if tiles:
        bricks = tile_exposed_tops(bricks, voxels)
    return Build(bricks, model, grid, mirrored, vocabulary, fixtures)
