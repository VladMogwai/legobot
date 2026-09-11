"""3D-меш -> воксельная сетка.

Воксель = 1 штырёк по горизонтали и 1 кирпич по высоте.
Ось Z исходной модели считается вертикалью.
"""
import numpy as np
import trimesh

from .parts import BRICK_ASPECT


SYMMETRY_TOLERANCE = 0.05  # доля вокселей, не совпавших с отражением, при которой модель ещё считается симметричной


def voxelize_mesh(path: str, grid: int) -> np.ndarray:
    """Возвращает bool-массив [x, y, z], z — вверх. `grid` — штырьков по длинной стороне.

    Массив выровнен так, что середина модели по X совпадает с серединой массива:
    отражение массива — это отражение модели.
    """
    mesh = trimesh.load(path, force="mesh")
    # Сжимаем по вертикали, чтобы кубический воксель соответствовал пропорциям кирпича.
    mesh.apply_scale([1.0, 1.0, 1.0 / BRICK_ASPECT])
    pitch = mesh.extents[:2].max() / grid
    vox = mesh.voxelized(pitch)
    center_x = mesh.bounds[:, 0].mean()
    center_index = (center_x - vox.transform[0, 3]) / pitch
    return _align_mirror_axis(vox.fill().matrix, center_index)


def _align_mirror_axis(voxels: np.ndarray, center_index: float) -> np.ndarray:
    """Дополняет массив пустыми колонками по X, чтобы ось симметрии легла в его середину."""
    target = round(2 * center_index) / 2          # ближайшая клетка или граница между клетками
    current = (voxels.shape[0] - 1) / 2
    pad = int(round(2 * abs(target - current)))
    if pad == 0:
        return voxels
    left, right = (0, pad) if target > current else (pad, 0)
    return np.pad(voxels, ((left, right), (0, 0), (0, 0)))


def is_symmetric(voxels: np.ndarray) -> bool:
    """Симметрична ли модель относительно плоскости, перпендикулярной X."""
    mismatch = (voxels ^ np.flip(voxels, axis=0)).sum()
    return mismatch / max(voxels.sum(), 1) <= SYMMETRY_TOLERANCE


def symmetrize(voxels: np.ndarray) -> np.ndarray:
    """Выравнивает модель по зеркалу: воксель есть, если он есть с любой из сторон."""
    return voxels | np.flip(voxels, axis=0)
