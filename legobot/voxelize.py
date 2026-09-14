"""3D-меш -> воксельная модель с цветом.

Воксель = 1 штырёк по горизонтали и 1 кирпич по высоте.
Ось Z исходной модели считается вертикалью.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


SYMMETRY_TOLERANCE = 0.02  # 90-й перцентиль расстояния от отражённой поверхности до меша, в долях ширины


@dataclass
class VoxelModel:
    occupancy: np.ndarray          # bool [x, y, z], z — вверх
    colors: np.ndarray | None      # uint8 [x, y, z, 3]; None — меш без цвета
    symmetric: bool                # меш симметричен относительно плоскости, перпендикулярной X

    @property
    def shape(self):
        return self.occupancy.shape


def voxelize_mesh(path: str, grid: int, aspect: float, force_symmetric: bool = False) -> VoxelModel:
    """`grid` — штырьков по длинной стороне; `aspect` — высота слоя в долях шага штырьков.
    force_symmetric — считать модель симметричной по лучшей оси, даже если меш кривоват (фото).

    Массив выровнен так, что середина модели по X совпадает с серединой массива:
    отражение массива — это отражение модели.
    """
    mesh = trimesh.load(path, force="mesh")
    symmetric = _orient_mirror_axis(mesh, force_symmetric)
    # Масштабируем по вертикали, чтобы кубический воксель соответствовал пропорциям детали.
    mesh.apply_scale([1.0, 1.0, 1.0 / aspect])
    pitch = mesh.extents[:2].max() / grid
    vox = mesh.voxelized(pitch).fill()
    occupancy = vox.matrix
    colors = _sample_colors(mesh, vox, occupancy)

    center_x = mesh.bounds[:, 0].mean()
    center_index = (center_x - vox.transform[0, 3]) / pitch
    model = VoxelModel(occupancy, colors, symmetric)
    return _align_mirror_axis(model, center_index)


def _orient_mirror_axis(mesh, force: bool = False) -> bool:
    """Находит плоскость симметрии: перебирает поворот вокруг вертикали (нейросетевые меши
    часто повёрнуты к осям на десятки градусов), затем разворачивает меш так, чтобы плоскость
    симметрии стала перпендикулярна X. Возвращает, симметричен ли меш вообще."""
    yaw, axis, error = _best_mirror_alignment(mesh)
    if error > SYMMETRY_TOLERANCE and not force:
        return False
    total = yaw + (90.0 if axis == 1 else 0.0)
    if total:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(total), [0, 0, 1]))
    return True


def _best_mirror_alignment(mesh) -> tuple[float, int, float]:
    """(поворот в градусах, ось 0|1, ошибка): поворот вокруг Z, при котором отражение
    относительно серединной плоскости, перпендикулярной оси, ложится на поверхность лучше всего."""
    surface, _ = trimesh.sample.sample_surface(mesh, 150_000, seed=0)
    probe, _ = trimesh.sample.sample_surface(mesh, 4_000, seed=1)

    def error_at(yaw: float) -> tuple[float, int]:
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        surf, prb = surface @ rot.T, probe @ rot.T
        tree = cKDTree(surf)
        best = (np.inf, 0)
        for axis in (0, 1):
            mirrored = prb.copy()
            center = (surf[:, axis].min() + surf[:, axis].max()) / 2
            mirrored[:, axis] = 2 * center - mirrored[:, axis]
            distance, _ = tree.query(mirrored)
            err = float(np.percentile(distance, 90) / np.ptp(surf[:, axis]))
            best = min(best, (err, axis))
        return best

    coarse = min(((*error_at(y), y) for y in range(-45, 46, 5)), key=lambda t: t[0])
    fine = min(((*error_at(y), y) for y in np.arange(coarse[2] - 4, coarse[2] + 4.1, 1.0)), key=lambda t: t[0])
    err, axis, yaw = fine
    return float(yaw), int(axis), err


SURFACE_SAMPLES = 400_000


def _sample_colors(mesh, vox, occupancy):
    """Поверхностные воксели берут цвет ближайшей точки поверхности, внутренние — основной цвет.

    Точки насыпаются по поверхности равномерно, а не берутся из вершин: у мешей вершины
    сгущаются на мелких деталях (канавки, швы), и ближайшая вершина врёт про цвет."""
    vertex_colors = _vertex_colors(mesh)
    if vertex_colors is None:
        return None
    points, face_ids = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)
    point_colors = vertex_colors[mesh.faces[face_ids]].mean(axis=1).astype(np.uint8)
    surface = occupancy & ~interior(occupancy)
    idx = np.argwhere(surface)
    centers = trimesh.transform_points(idx.astype(float), vox.transform)
    _, nearest = cKDTree(points).query(centers)
    sampled = point_colors[nearest]
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
    """Воксели, со всех шести сторон закрытые другими вокселями: снаружи их не видно.
    Края массива считаются пустыми (без заворачивания, как у np.roll)."""
    padded = np.pad(occupancy, 1)
    inner = occupancy.copy()
    for axis in range(3):
        before = np.roll(padded, 1, axis)[1:-1, 1:-1, 1:-1]
        after = np.roll(padded, -1, axis)[1:-1, 1:-1, 1:-1]
        inner &= before & after
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
    return VoxelModel(occupancy, colors, model.symmetric)


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
    return VoxelModel(occ | np.flip(occ, axis=0), colors, model.symmetric)


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
            return VoxelModel(keep, model.colors, model.symmetric)
        v = keep
