"""Сборка целиком: меш -> воксели -> кирпичи. Одна функция для CLI, подбора размера и бота."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .colors import nearest_codes, recolor
from .finish import tile_exposed_tops
from .fixtures import WHEEL_BY_PART, Fixture, wheel_fixtures
from .layout import ANY_COLOR, EXPOSED_TOP, PlacedBrick, layout_bricks
from .parts import STUD_LDU, Vocabulary
from .slopes import PlacedSlope, find_slopes
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
    slopes: list[PlacedSlope]

    @property
    def layer_mm(self) -> float:
        return STUD_MM * self.vocabulary.height / STUD_LDU

    @property
    def part_count(self) -> int:
        return len(self.bricks) + len(self.slopes) + len(self.fixtures)

    @property
    def size_mm(self) -> tuple[float, float, float]:
        """Габариты в миллиметрах: ширина (X), длина (Z), высота."""
        occ = self.model.occupancy
        x, z, k = (np.ptp(np.nonzero(occ)[i]) + 1 for i in range(3))
        return x * STUD_MM, z * STUD_MM, k * self.layer_mm

    @property
    def colors(self) -> Counter:
        return Counter(b.color for b in self.bricks)


def _despeckle(codes: np.ndarray, surface: np.ndarray, passes: int = 1) -> np.ndarray:
    """Поверхностный воксель, цвет которого не поддерживает большинство соседей по поверхности,
    перекрашивается в цвет большинства. Убирает крапины от теней и шума текстуры."""
    codes = codes.copy()
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for _ in range(passes):
        changed = codes.copy()
        for x, y, z in np.argwhere(surface):
            votes = Counter()
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < codes.shape[0] and 0 <= ny < codes.shape[1] and 0 <= nz < codes.shape[2] and surface[nx, ny, nz]:
                    votes[codes[nx, ny, nz]] += 1
            if votes:
                top, n = votes.most_common(1)[0]
                if top != codes[x, y, z] and n > sum(votes.values()) * 0.6:
                    changed[x, y, z] = top
        codes = changed
    return codes


def build(mesh_path: str, grid: int, default_color: int, vocabulary: Vocabulary,
          wheels: str | None = None, tiles: bool = True, max_colors: int = 4,
          force_symmetric: bool = False, recolor_map: dict[int, int] | None = None,
          slopes: bool = False) -> Build:
    """wheels: None — без колёс, "auto" — подобрать по арке, иначе номер детали колеса.
    max_colors — до скольких цветов сводить цвет меша. force_symmetric — зеркалить даже кривой меш.
    slopes — закрывать ступеньки скосами (только при кладке пластинами)."""
    model = voxelize_mesh(mesh_path, grid, vocabulary.aspect, force_symmetric)
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
        codes[surface] = nearest_codes(model.colors[surface], max_colors)
        codes = _despeckle(codes, surface)
        if recolor_map:
            codes[surface] = recolor(codes[surface], recolor_map)
        body_color = Counter(codes[surface].tolist()).most_common(1)[0][0]

    placed_slopes: list[PlacedSlope] = []
    fixed = None
    if slopes:
        assert vocabulary.name == "plates", "скосы рассчитаны на кладку пластинами"
        placed_slopes, fixed = find_slopes(voxels, codes, body_color, mirrored, ANY_COLOR, model.normals)

    if tiles:
        exposed = voxels.copy()
        exposed[:, :, :-1] &= ~voxels[:, :, 1:]
        codes[exposed] = np.where(codes[exposed] == ANY_COLOR, body_color, codes[exposed]) | EXPOSED_TOP
    bricks = layout_bricks(voxels, codes, body_color, vocabulary, mirrored=mirrored, fixed=fixed)
    covered = sum(b.part.area for b in bricks) + (int((fixed >= 0).sum()) if fixed is not None else 0)
    assert covered == int(voxels.sum()), f"покрыто {covered} из {int(voxels.sum())} вокселей"
    if tiles:
        bricks = tile_exposed_tops(bricks, voxels)
    return Build(bricks, model, grid, mirrored, vocabulary, fixtures, placed_slopes)
