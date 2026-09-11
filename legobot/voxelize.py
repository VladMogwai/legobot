"""3D-меш -> воксельная модель с цветом.

Воксель = 1 штырёк по горизонтали и 1 кирпич по высоте.
Ось Z исходной модели считается вертикалью.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .parts import BRICK_ASPECT

SYMMETRY_TOLERANCE = 0.05  # доля вокселей, не совпавших с отражением, при которой модель ещё считается симметричной


@dataclass
class VoxelModel:
    occupancy: np.ndarray          # bool [x, y, z], z — вверх
    colors: np.ndarray | None      # uint8 [x, y, z, 3]; None — меш без цвета

    @property
    def shape(self):
        return self.occupancy.shape


def voxelize_mesh(path: str, grid: int) -> VoxelModel:
    """`grid` — штырьков по длинной стороне.

    Массив выровнен так, что середина модели по X совпадает с серединой массива:
    отражение массива — это отражение модели.
    """
    mesh = trimesh.load(path, force="mesh")
    # Сжимаем по вертикали, чтобы кубический воксель соответствовал пропорциям кирпича.
    mesh.apply_scale([1.0, 1.0, 1.0 / BRICK_ASPECT])
    pitch = mesh.extents[:2].max() / grid
    vox = mesh.voxelized(pitch).fill()
    occupancy = vox.matrix
    colors = _sample_colors(mesh, vox, occupancy)

    center_x = mesh.bounds[:, 0].mean()
    center_index = (center_x - vox.transform[0, 3]) / pitch
    return _align_mirror_axis(VoxelModel(occupancy, colors), center_index)


def _sample_colors(mesh, vox, occupancy):
    """Поверхностные воксели берут цвет ближайшей вершины меша, внутренние — основной цвет поверхности."""
    vertex_colors = _vertex_colors(mesh)
    if vertex_colors is None:
        return None
    surface = occupancy & ~interior(occupancy)
    idx = np.argwhere(surface)
    centers = trimesh.transform_points(idx.astype(float), vox.transform)
    _, nearest = cKDTree(mesh.vertices).query(centers)
    sampled = vertex_colors[nearest]
    colors = np.zeros(occupancy.shape + (3,), dtype=np.uint8)
    colors[occupancy] = _dominant(sampled)
    colors[tuple(idx.T)] = sampled
    return colors


def _vertex_colors(mesh):
    visual = mesh.visual
    if hasattr(visual, "to_color"):  # текстура -> цвета вершин
        visual = visual.to_color()
    rgb = np.asarray(visual.vertex_colors)[:, :3]
    if len(np.unique(rgb, axis=0)) <= 1:
        return None  # у меша нет цвета (STL и т.п.)
    return rgb


def interior(occupancy: np.ndarray) -> np.ndarray:
    """Воксели, со всех шести сторон закрытые другими вокселями: снаружи их не видно."""
    inner = occupancy.copy()
    for axis in range(3):
        inner &= np.roll(occupancy, 1, axis) & np.roll(occupancy, -1, axis)
    return inner


def _dominant(rgb):
    values, counts = np.unique(rgb, axis=0, return_counts=True)
    return values[counts.argmax()]


def _align_mirror_axis(model: VoxelModel, center_index: float) -> VoxelModel:
    """Дополняет массив пустыми колонками по X, чтобы ось симметрии легла в его середину."""
    target = round(2 * center_index) / 2          # ближайшая клетка или граница между клетками
    current = (model.shape[0] - 1) / 2
    pad = int(round(2 * abs(target - current)))
    if pad == 0:
        return model
    left, right = (0, pad) if target > current else (pad, 0)
    occupancy = np.pad(model.occupancy, ((left, right), (0, 0), (0, 0)))
    colors = None if model.colors is None else np.pad(model.colors, ((left, right), (0, 0), (0, 0), (0, 0)))
    return VoxelModel(occupancy, colors)


def is_symmetric(model: VoxelModel) -> bool:
    """Симметрична ли модель относительно плоскости, перпендикулярной X."""
    occ = model.occupancy
    mismatch = (occ ^ np.flip(occ, axis=0)).sum()
    return mismatch / max(occ.sum(), 1) <= SYMMETRY_TOLERANCE


def symmetrize(model: VoxelModel) -> VoxelModel:
    """Выравнивает модель по зеркалу: воксель есть, если он есть с любой из сторон;
    цвет правой половины — отражение левой."""
    occ = model.occupancy
    colors = model.colors
    if colors is not None:
        mirrored = np.flip(colors, axis=0)
        # Воксель, появившийся только из отражения, берёт цвет отражения.
        colors = np.where(occ[..., None], colors, mirrored)
        half = occ.shape[0] // 2
        colors[half:] = np.flip(colors, axis=0)[half:]
    return VoxelModel(occ | np.flip(occ, axis=0), colors)


def drop_floating(model: VoxelModel) -> VoxelModel:
    """Убирает воксели, у которых пусто и снизу, и сверху: их не к чему прикрепить."""
    v = model.occupancy.copy()
    while True:
        below = np.zeros_like(v)
        below[:, :, 1:] = v[:, :, :-1]
        below[:, :, 0] = True  # земля
        above = np.zeros_like(v)
        above[:, :, :-1] = v[:, :, 1:]
        keep = v & (below | above)
        if keep.sum() == v.sum():
            return VoxelModel(keep, model.colors)
        v = keep
