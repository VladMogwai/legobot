"""Режим мозаики: пиксельная фигура -> пиксель = деталь.

Два варианта сборки. Стоячая (по умолчанию): пиксель — кирпич глубиной STANDING_DEPTH
штырьков, ряд пикселей — слой кладки, одноцветные соседи в ряду сливаются в 2×4, 2×6, 2×8;
фигурка стоит сама, как Pixel Pals. Плоская: подложка из пластин по силуэту и тайлы 1x1.

Из 3D-файла: берём переднюю грань меша (тонкая ось — нормаль), растеризуем её, находим шаг
пиксельной сетки по периодичности границ цвета, снимаем цвет в центре каждой клетки.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

from .colors import nearest_codes
from .finish import TILES
from .layout import ANY_COLOR, PlacedBrick, layout_bricks
from .colors import code_by_name
from .parts import BRICKS, PLATES
from .preferences import preferences

RASTER = 512
STANDING_DEPTH = 3  # штырьков в глубину у стоячей фигурки: 2 пикселя + 1 стенка (через ряд 1 + 2)
SAMPLES = 1_500_000
MIN_PITCH, MAX_PITCH = 6, 80  # шаг сетки в пикселях растра
BLACK = 0


@dataclass
class Mosaic:
    codes: np.ndarray   # [W, H] коды LDraw, -1 — пусто; H — сверху вниз
    pitch_px: float
    width: int
    height: int


def mosaic_from_mesh(mesh_path: str, max_colors: int = 9, pixels_wide: int | None = None) -> Mosaic:
    mesh = trimesh.load(mesh_path, force="mesh")
    image, present, span = _front_face(mesh)
    pitch = span[0] / (span.max()) * RASTER / pixels_wide if pixels_wide else _detect_pitch(image, present)
    phase_x, phase_z = _phase(image, present, pitch)
    colors, mask = _sample_cells(image, present, pitch, phase_x, phase_z)
    codes = np.full(mask.shape, -1)
    codes[mask] = nearest_codes(colors[mask], max_colors)
    return Mosaic(codes, pitch, *mask.shape)


def _front_face(mesh):
    """Растр передней грани: цвета усреднены по точкам поверхности, дыры сглажены медианой."""
    thin = int(np.argmin(mesh.extents))
    axes = [a for a in range(3) if a != thin]
    points, face_ids = trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)
    normals = mesh.face_normals[face_ids]
    colors = np.asarray(mesh.visual.vertex_colors)[mesh.faces[face_ids]][:, :, :3].mean(1)
    facing = normals[:, thin] < -0.8
    if facing.sum() < (normals[:, thin] > 0.8).sum():
        facing = normals[:, thin] > 0.8
    p, c = points[facing][:, axes], colors[facing]
    lo, span = p.min(0), p.max(0) - p.min(0)
    ij = ((p - lo) / span.max() * (RASTER - 1)).astype(int)
    image = np.zeros((RASTER, RASTER, 3)); count = np.zeros((RASTER, RASTER))
    np.add.at(image, (ij[:, 0], ij[:, 1]), c); np.add.at(count, (ij[:, 0], ij[:, 1]), 1)
    present = count > 0
    image = np.where(present[..., None], image / np.maximum(count[..., None], 1), 0)
    image = ndimage.median_filter(image, size=(5, 5, 1))
    present = ndimage.binary_closing(present, iterations=2)
    return image, present, span


def _edge_profile(image, axis):
    grad = np.abs(np.diff(image, axis=axis)).sum(2)
    return grad.sum(1 - axis)


def _detect_pitch(image, present) -> float:
    """Шаг сетки — первый выраженный пик автокорреляции профиля границ цвета (по ширине)."""
    profile = _edge_profile(image, 0)
    s = profile - profile.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    peaks = [(ac[l], l) for l in range(MIN_PITCH, MAX_PITCH) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
    if not peaks:
        raise ValueError("не найден шаг пиксельной сетки — задайте --pixels")
    return float(max(peaks)[1])


def _phase(image, present, pitch):
    """Смещение сетки по каждой оси: линии сетки должны попадать на пики границ цвета."""
    phases = []
    for axis in (0, 1):
        profile = _edge_profile(image, axis)
        best = max(range(int(pitch)), key=lambda ph: profile[ph::int(round(pitch))].sum())
        phases.append(best)
    return phases


def _sample_cells(image, present, pitch, phase_x, phase_z):
    nx = int((RASTER - phase_x) // pitch) + 1
    nz = int((RASTER - phase_z) // pitch) + 1
    colors = np.zeros((nx, nz, 3), dtype=np.uint8); mask = np.zeros((nx, nz), dtype=bool)
    inner = pitch * 0.25
    for i in range(nx):
        for j in range(nz):
            x0, z0 = phase_x + i * pitch, phase_z + j * pitch
            xs = slice(int(x0 + inner), int(x0 + pitch - inner)); zs = slice(int(z0 + inner), int(z0 + pitch - inner))
            cell_present = present[xs, zs]
            if cell_present.size == 0 or cell_present.mean() < 0.5:
                continue
            mask[i, j] = True
            colors[i, j] = np.median(image[xs, zs][cell_present], axis=0)
    return colors, mask


def mosaic_bricks(mosaic: Mosaic, base_color: int = BLACK) -> list[PlacedBrick]:
    """Слой 0 — подложка из пластин по силуэту, слой 1 — тайлы 1x1 по цветам."""
    mask = mosaic.codes >= 0
    voxels = mask[:, :, None]
    codes = np.full(voxels.shape, ANY_COLOR)
    base = layout_bricks(voxels, codes, base_color, PLATES)
    tile = TILES[(1, 1)]
    tiles = [PlacedBrick(tile, x, z, 1, rotated=False, color=int(mosaic.codes[x, z]))
             for x, z in np.argwhere(mask)]
    return base + tiles


def standing_bricks(mosaic: Mosaic) -> list[PlacedBrick]:
    """Стоячая фигурка. Спереди пиксели, сзади стенка (цвет — в legobot.toml; у Pixel Pals чёрная).
    Глубина STANDING_DEPTH; граница «пиксель/стенка» чередуется по рядам (2+1, 1+2), чтобы
    стенка и пиксели связывались штырьками через ряд, иначе это две несвязанные стены.
    Выступы без опоры (ухо, край ноги) подпираются столбиком стенки до ближайшей опоры."""
    mask = mosaic.codes >= 0
    xs, ys = np.nonzero(mask)
    mask = mask[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
    codes = mosaic.codes[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
    mask, codes = np.flip(mask, axis=1), np.flip(codes, axis=1)      # слой 0 — нижний ряд
    back = mask | _pillars(mask, codes)
    back_color = code_by_name(preferences()["mosaic"]["back_color"])
    nx, ny = mask.shape
    voxels = np.zeros((nx, STANDING_DEPTH, ny), dtype=bool)
    colors = np.full(voxels.shape, ANY_COLOR)
    for k in range(ny):
        front_depth = STANDING_DEPTH - 1 if k % 2 == 0 else 1
        voxels[:, :front_depth, k] = mask[:, k, None]
        colors[:, :front_depth, k] = codes[:, k, None]
        voxels[:, front_depth:, k] |= back[:, k, None]
        colors[:, front_depth:, k] = np.where(back[:, k, None], back_color, ANY_COLOR)
    return layout_bricks(voxels, colors, back_color, BRICKS)


def _pillars(mask: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Столбики стенки только под кусками, не связанными штырьками с основным телом.
    Связь — вертикальное соседство (кирпич держится и снизу, и сверху) или одноцветный сосед
    в ряду (такие сливаются в один кирпич). Висящая на плече рука — связана, подпорка не нужна."""
    nx, ny = mask.shape
    labels, _ = ndimage.label(mask, structure=[[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for x in range(nx - 1):
        for y in range(ny):
            if mask[x, y] and mask[x + 1, y] and codes[x, y] == codes[x + 1, y]:
                parent[find(labels[x, y])] = find(labels[x + 1, y])
    roots = np.array([[find(labels[x, y]) if mask[x, y] else -1 for y in range(ny)] for x in range(nx)])
    main = Counter(roots[mask].tolist()).most_common(1)[0][0]
    pillars = np.zeros_like(mask)
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y] or roots[x, y] == main:
                continue
            below = [yy for yy in range(y - 1, -1, -1) if mask[x, yy] and roots[x, yy] == main]
            above = [yy for yy in range(y + 1, ny) if mask[x, yy] and roots[x, yy] == main]
            if not below and not above:
                continue
            gap_down = y - below[0] if below else ny
            gap_up = above[0] - y if above else ny
            lo, hi = (below[0] + 1, y) if gap_down <= gap_up else (y + 1, above[0])
            pillars[x, lo:hi] = True
    return pillars
