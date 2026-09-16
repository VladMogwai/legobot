"""Пиксельная фигурка прямо из фото, без 3D: выправить перспективу по линиям сетки, снять
цвет с каждой клетки.

Фон вырезает rembg. Швы между пикселями дают два семейства прямых; их точки схода задают
гомографию, после которой сетка становится прямоугольной. Шаг сетки по каждой оси — из
автокорреляции профиля градиента, фаза — где линии сетки. Фигурка объёмная, и сбоку видна
её боковая грань из чёрного пластика; она заметно темнее напечатанного чёрного контура
спрайта (тот тёмно-серый), поэтому отделяется порогом яркости.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import color, feature, morphology, transform

from .colors import nearest_codes
from .mosaic import Mosaic

RECTIFIED_PX = 900          # длинная сторона выправленного изображения
MIN_PITCH, MAX_PITCH = 10, 80
SIDE_BRIGHTNESS = 0.12      # яркость (0..1), ниже которой клетка — боковая грань, а не пиксель


def mosaic_from_photo(image_path: str, max_colors: int = 12) -> Mosaic:
    rgb, mask = _cutout(image_path)
    rect, rmask = _rectify(rgb, mask)
    pitch_x, pitch_y, phase_x, phase_y = _grid(rect, rmask)
    colors, present = _sample_cells(rect, rmask, pitch_x, pitch_y, phase_x, phase_y)
    present &= _front_face(colors, present)
    codes = np.full(present.shape, -1)
    codes[present] = nearest_codes(colors[present], max_colors)
    return Mosaic(codes, (pitch_x + pitch_y) / 2, *present.shape)


def _cutout(path):
    import rembg
    rgba = np.asarray(rembg.remove(Image.open(path).convert("RGB")))
    return rgba[..., :3].astype(float) / 255, rgba[..., 3] > 128


def _rectify(rgb, mask):
    """Гомография по точкам схода двух семейств линий сетки; отражение убирается."""
    gray = color.rgb2gray(rgb)
    edges = feature.canny(gray, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    edges &= morphology.erosion(mask, morphology.disk(3))
    segs = np.array(transform.probabilistic_hough_line(edges, threshold=8, line_length=25, line_gap=2, rng=0), dtype=float)
    angle = np.degrees(np.arctan2(segs[:, 1, 1] - segs[:, 0, 1], segs[:, 1, 0] - segs[:, 0, 0])) % 180
    horizontal = (angle < 20) | (angle > 175)
    vertical = (angle > 70) & (angle < 110)
    if horizontal.sum() < 5 or vertical.sum() < 5:
        raise ValueError("не нашёл линии сетки на фото")
    center = np.array([rgb.shape[1] / 2, rgb.shape[0] / 2, 1.0])
    H = np.linalg.inv(np.c_[_vanishing_point(segs[horizontal]), _vanishing_point(segs[vertical]), center])
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys, np.ones(len(xs))] @ H.T
    pts = pts[:, :2] / pts[:, 2:3]
    lo, hi = pts.min(0), pts.max(0)
    scale = RECTIFIED_PX / (hi - lo).max()
    flip = np.linalg.det(_jacobian(H, center)) < 0       # гомография отразила картинку
    fit = np.array([[-scale if flip else scale, 0, (hi[0] if flip else -lo[0]) * scale + 20],
                    [0, scale, -lo[1] * scale + 20], [0, 0, 1]])
    tf = transform.ProjectiveTransform(matrix=fit @ H)
    shape = (int((hi - lo)[1] * scale + 40), int((hi - lo)[0] * scale + 40))
    rect = transform.warp(rgb, tf.inverse, output_shape=shape)
    rmask = transform.warp(mask.astype(float), tf.inverse, output_shape=shape) > 0.5
    return rect, rmask


def _vanishing_point(segs):
    """Точка, ближайшая ко всем прямым семейства (взвешено длиной отрезков)."""
    p0 = np.c_[segs[:, 0], np.ones(len(segs))]
    p1 = np.c_[segs[:, 1], np.ones(len(segs))]
    lines = np.cross(p0, p1)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lines *= np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)[:, None]
    return np.linalg.svd(lines)[2][-1]


def _jacobian(H, p):
    q = H @ p
    return (H[:2, :2] * q[2] - np.outer(q[:2], H[2, :2])) / q[2] ** 2


def _grid(rect, rmask):
    gray = color.rgb2gray(rect)
    profiles = [
        (np.abs(np.diff(gray, axis=1)) * rmask[:, :-1]).sum(0),   # вдоль x
        (np.abs(np.diff(gray, axis=0)) * rmask[:-1, :]).sum(1),   # вдоль y
    ]
    out = []
    for profile in profiles:
        s = profile - profile.mean()
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac /= ac[0]
        peaks = [(ac[l], l) for l in range(MIN_PITCH, MAX_PITCH) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
        if not peaks:
            raise ValueError("не нашёл шаг пиксельной сетки")
        l = max(peaks)[1]
        a, b, c = ac[l - 1], ac[l], ac[l + 1]
        pitch = l + 0.5 * (a - c) / (a - 2 * b + c)              # субпиксельно, по параболе
        n = int((len(profile) - MAX_PITCH) / pitch)
        phase = max(range(int(pitch)), key=lambda ph: sum(profile[int(round(ph + i * pitch))] for i in range(n)))
        out.append((pitch, phase))
    (pitch_x, phase_x), (pitch_y, phase_y) = out
    return pitch_x, pitch_y, phase_x, phase_y


def _sample_cells(rect, rmask, pitch_x, pitch_y, phase_x, phase_y):
    """Цвет — медиана внутренней части клетки; клетка есть, если маска покрывает её большую часть."""
    nx = int((rect.shape[1] - phase_x) / pitch_x)
    ny = int((rect.shape[0] - phase_y) / pitch_y)
    colors = np.zeros((nx, ny, 3), dtype=np.uint8)
    present = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            x0, y0 = phase_x + i * pitch_x, phase_y + j * pitch_y
            xs = slice(int(x0 + pitch_x * 0.3), int(x0 + pitch_x * 0.7))
            ys = slice(int(y0 + pitch_y * 0.3), int(y0 + pitch_y * 0.7))
            if rmask[ys, xs].mean() < 0.6:
                continue
            present[i, j] = True
            colors[i, j] = (np.median(rect[ys, xs].reshape(-1, 3), axis=0) * 255).astype(np.uint8)
    return colors, present


def _front_face(colors, present):
    """Клетки передней грани: всё, что не чёрный пластик боковой грани, одним связным куском
    (обрывки по краям — блики на боковой грани)."""
    from scipy import ndimage
    front = present & (color.rgb2gray(colors / 255.0) >= SIDE_BRIGHTNESS)
    labels, n = ndimage.label(front)
    if n > 1:
        sizes = ndimage.sum(front, labels, range(1, n + 1))
        front = labels == (int(np.argmax(sizes)) + 1)
    return front
