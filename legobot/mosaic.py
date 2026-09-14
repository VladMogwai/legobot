"""Режим мозаики: плоская пиксельная фигура -> один пиксель = один тайл 1x1.

Берём переднюю грань меша (тонкая ось — нормаль), растеризуем её, находим шаг
пиксельной сетки по периодичности границ цвета (в пиксель-арте цвет меняется только
на линиях сетки), снимаем цвет в центре каждой клетки. Сборка: подложка из пластин
по силуэту и слой тайлов 1x1 по цветам.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

from .colors import nearest_codes
from .finish import TILES
from .layout import ANY_COLOR, PlacedBrick, layout_bricks
from .parts import PLATES

RASTER = 512
SAMPLES = 1_500_000
MIN_PITCH, MAX_PITCH = 6, 80  # шаг сетки в пикселях растра
BLACK = 0


@dataclass
class Mosaic:
    codes: np.ndarray   # [W, H] коды LDraw, -1 — пусто
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
