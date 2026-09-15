"""3D-меш -> воксельная модель с цветом.

Воксель = 1 штырёк по горизонтали и 1 кирпич по высоте.
Ось Z исходной модели считается вертикалью.
"""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


SYMMETRY_TOLERANCE = 0.02  # 90-й перцентиль расстояния от отражённой поверхности до меша, в долях ширины
SUBSAMPLE = 3  # подвокселей на ребро при оценке заполнения


@dataclass
class VoxelModel:
    occupancy: np.ndarray          # bool [x, y, z], z — вверх
    colors: np.ndarray | None      # uint8 [x, y, z, 3]; None — меш без цвета
    symmetric: bool                # меш симметричен относительно плоскости, перпендикулярной X
    normals: np.ndarray | None = None   # float32 [x, y, z, 3], нормаль поверхности у поверхностных вокселей
                                        # (в штырьках по всем осям, z вверх), нули — внутри

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
    occupancy, origin = _occupancy(mesh, pitch)
    colors = _sample_colors(mesh, occupancy, origin, pitch)
    normals = _sample_normals(mesh, occupancy, origin, pitch, aspect)

    center_x = mesh.bounds[:, 0].mean()
    center_index = (center_x - origin[0]) / pitch
    model = VoxelModel(occupancy, colors, symmetric, normals)
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


def _occupancy(mesh, pitch):
    """Воксель занят, если внутри меша не меньше половины его объёма.

    trimesh.voxelized помечает любой воксель, которого поверхность хоть краешком коснулась:
    форма раздувается на полвокселя и покрывается случайными буграми. Поэтому считаем на
    сетке в SUBSAMPLE раз мельче и укрупняем по доле заполнения. Возвращает (occupancy, origin):
    origin — центр вокселя [0, 0, 0] в координатах меша."""
    n = SUBSAMPLE
    fine = mesh.voxelized(pitch / n).fill()
    m = fine.matrix
    pad = [(0, (-s) % n) for s in m.shape]
    m = np.pad(m, pad)
    blocks = m.reshape(m.shape[0] // n, n, m.shape[1] // n, n, m.shape[2] // n, n)
    occupancy = blocks.sum(axis=(1, 3, 5)) >= n ** 3 / 2
    origin = fine.transform[:3, 3] + (n - 1) / 2 * pitch / n
    return occupancy, origin


def _sample_colors(mesh, occupancy, origin, pitch):
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
    centers = origin + idx * pitch
    _, nearest = cKDTree(points).query(centers)
    sampled = point_colors[nearest]
    colors = np.zeros(occupancy.shape + (3,), dtype=np.uint8)
    colors[occupancy] = _dominant(sampled)
    colors[tuple(idx.T)] = sampled
    return colors


def _sample_normals(mesh, occupancy, origin, pitch, aspect):
    """Средняя нормаль меша в каждом поверхностном вокселе (по точкам поверхности, попавшим в него;
    если не попало ни одной — по ближайшей). Меш растянут по вертикали в 1/aspect раз, поэтому
    вертикальную компоненту возвращаем в исходный масштаб."""
    points, face_ids = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)
    normals = mesh.face_normals[face_ids].copy()
    normals[:, 2] /= aspect
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    idx = np.round((points - origin) / pitch).astype(int)
    inside = np.all((idx >= 0) & (idx < occupancy.shape), axis=1)
    summed = np.zeros(occupancy.shape + (3,))
    np.add.at(summed, tuple(idx[inside].T), normals[inside])
    surface = occupancy & ~interior(occupancy)
    cells = np.argwhere(surface)
    empty = np.linalg.norm(summed[tuple(cells.T)], axis=1) < 1e-9
    if empty.any():
        _, nearest = cKDTree(points).query(origin + cells[empty] * pitch)
        summed[tuple(cells[empty].T)] = normals[nearest]
    out = np.zeros(occupancy.shape + (3,), dtype=np.float32)
    n = summed[tuple(cells.T)]
    out[tuple(cells.T)] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return out


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
    pad4 = lambda a: None if a is None else np.pad(a, ((left, right), (0, 0), (0, 0), (0, 0)))
    occupancy = np.pad(model.occupancy, ((left, right), (0, 0), (0, 0)))
    return VoxelModel(occupancy, pad4(model.colors), model.symmetric, pad4(model.normals))


def symmetrize(model: VoxelModel) -> VoxelModel:
    """Выравнивает модель по зеркалу: воксель есть, если он есть с любой из сторон;
    цвет правой половины — отражение левой."""
    occ = model.occupancy
    half = occ.shape[0] // 2

    def mirror(field, flip_x=False):
        if field is None:
            return None
        mirrored = np.flip(field, axis=0)
        if flip_x:
            mirrored = mirrored * np.array([-1, 1, 1], dtype=field.dtype)
        # Воксель, появившийся только из отражения, берёт значение отражения.
        field = np.where(occ[..., None], field, mirrored)
        field[half:] = np.flip(field, axis=0)[half:] * (np.array([-1, 1, 1], dtype=field.dtype) if flip_x else 1)
        return field

    return VoxelModel(occ | np.flip(occ, axis=0), mirror(model.colors), model.symmetric, mirror(model.normals, flip_x=True))


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
            return VoxelModel(keep, model.colors, model.symmetric, model.normals)
        v = keep
