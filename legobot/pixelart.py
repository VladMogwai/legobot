"""Пиксельная фигурка прямо из фото, без 3D: выправить перспективу по линиям сетки, снять
цвет с каждой клетки.

Фон вырезает rembg. Швы между пикселями дают два семейства прямых; их точки схода задают
гомографию, после которой сетка становится прямоугольной. Шаг сетки по каждой оси — из
автокорреляции профиля градиента, фаза — где линии сетки. Фигурка объёмная: сбоку и снизу
видны чёрный пластик боковин и основание. Их не отличить от чёрного контура спрайта по цвету,
зато отличает геометрия: силуэт = передняя грань, сдвинутая вдоль ребра на несколько клеток.
Сдвиг подбирается так, чтобы срезались только тёмные клетки, а новый край силуэта оставался
тёмным (контур спрайта): срез больше нужного обнажил бы цветные клетки.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import color, feature, morphology, transform

from .colors import flat_codes, studio_palette
from .mosaic import Mosaic
from .preferences import preferences

RECTIFIED_PX = 900          # длинная сторона выправленного изображения
MIN_PITCH, MAX_PITCH = 8, 80
FUNDAMENTAL_RATIO = 0.9           # пик автокорреляции не ниже 90 % от лучшего — тот же период
FLAT_BORDER_STD = 0.03            # рамка картинки одноцветная — это рисунок на ровном фоне, не фото
FLAT_BACKGROUND_TOLERANCE = 0.12  # насколько цвет должен отличаться от фона, чтобы быть объектом
DARK_L, DARK_CHROMA = 0.45, 0.15   # «тёмная и бесцветная» клетка: чёрный контур спрайта или чёрный пластик
MAX_SIDE = 4                       # боковая грань не шире стольких клеток
DOWNSCALE_COVERAGE = 0.35          # при укрупнении клетка есть, если объект занимает хотя бы столько её площади
BASE_MIN_WIDTH = 0.6               # подставка: сплошной тёмный нижний ряд не уже такой доли ширины фигурки


@dataclass
class PhotoMosaic:
    mosaic: Mosaic
    cell_colors: np.ndarray   # [W, H, 3] цвет клетки на фото (после калибровки уровней)


def mosaic_from_photo(image_path: str, max_colors: int = 16, keep_background: bool = False) -> PhotoMosaic:
    """keep_background — панно: у плоского рисунка фон выкладывается как цвет, а не отбрасывается."""
    rgb, mask = _cutout(image_path)
    if keep_background and is_flat(rgb, mask):
        mask = np.ones_like(mask)
    rect, rmask = _rectify(rgb, mask)
    pitch_x, pitch_y, phase_x, phase_y = _grid(rect, rmask)
    colors, present = _sample_cells(rect, rmask, pitch_x, pitch_y, phase_x, phase_y)
    front = present if keep_background else _front_face(colors, present)
    black, white = _anchors(colors, present, front)
    codes = np.full(present.shape, -1)
    prefs = preferences()["mosaic"]
    codes[front], calibrated = flat_codes(colors[front], max_colors, black, white,
                                          outline_black=prefs["outline_black"], common_only=prefs["palette"] == "common")
    cell_colors = np.zeros_like(colors)
    cell_colors[front] = calibrated
    return PhotoMosaic(Mosaic(codes, (pitch_x + pitch_y) / 2, *present.shape), cell_colors)


def mosaic_from_image(image_path: str, width: int, max_colors: int = 12, keep_background: bool = False,
                      outline: bool = False, contrast: bool = False, dither: bool = False) -> PhotoMosaic:
    """Любая картинка → пиксель-арт шириной `width` клеток. Объект (или вся картинка при
    keep_background) режется на клетки, цвет клетки — медиана её пикселей, потом ближайший
    пластик. Одиночные клетки без соседей убираются; outline — чёрный контур в одну клетку
    вокруг силуэта, как у Pixel Pals."""
    from scipy import ndimage
    rgb, mask = _cutout(image_path)
    if keep_background:
        mask = np.ones_like(mask)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cell = (x1 - x0) / width
    height = max(1, int(round((y1 - y0) / cell)))
    colors = np.zeros((width, height, 3), dtype=np.uint8)
    present = np.zeros((width, height), dtype=bool)
    for i in range(width):
        for j in range(height):
            cx0, cx1 = x0 + int(i * cell), x0 + max(int(i * cell) + 1, int((i + 1) * cell))
            cy0, cy1 = y0 + int(j * cell), y0 + max(int(j * cell) + 1, int((j + 1) * cell))
            m = mask[cy0:cy1, cx0:cx1]
            if m.size == 0 or m.mean() < DOWNSCALE_COVERAGE:
                continue
            present[i, j] = True
            # медиана, а не среднее: клетка «наполовину контур, наполовину заливка» становится
            # тем, чего в ней больше, а не серой кашей
            colors[i, j] = (np.median(rgb[cy0:cy1, cx0:cx1][m], axis=0) * 255).astype(np.uint8)
    neighbours = ndimage.convolve(present.astype(int), [[0, 1, 0], [1, 0, 1], [0, 1, 0]], mode="constant")
    present &= neighbours >= 2                                   # одиночные клетки и «усы» не собрать
    if outline:
        ring = ndimage.binary_dilation(present) & ~present
        colors[ring] = 0
        present |= ring
    if contrast:
        colors[present] = _vivid(colors[present])
    codes = np.full(present.shape, -1)
    if dither:
        codes = _dither(colors, present, common_only=preferences()["mosaic"]["palette"] == "common")
        calibrated = colors[present]
    else:
        codes[present], calibrated = flat_codes(colors[present], max_colors, None, None,
                                                outline_black=preferences()["mosaic"]["outline_black"] and not contrast,
                                                common_only=preferences()["mosaic"]["palette"] == "common")
        if contrast:
            codes[present] = _grey_tones(colors[present], codes[present])
    codes = _despeckle_codes(codes)
    cell_colors = np.zeros_like(colors)
    cell_colors[present] = calibrated
    return PhotoMosaic(Mosaic(codes, cell, width, height), cell_colors)


def _despeckle_codes(codes: np.ndarray) -> np.ndarray:
    """Клетка, у которой все четыре соседа одного цвета, а она другого, — крапина:
    перекрашиваем в цвет соседей. Одна крапина — одна лишняя деталь и шум в рисунке."""
    out = codes.copy()
    w, h = codes.shape
    for x in range(w):
        for y in range(h):
            if codes[x, y] < 0:
                continue
            around = [codes[x + dx, y + dy] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + dx < w and 0 <= y + dy < h]
            around = [c for c in around if c >= 0]
            if len(around) == 4 and len(set(around)) == 1 and around[0] != codes[x, y]:
                out[x, y] = around[0]
    return out


def _dither(colors: np.ndarray, present: np.ndarray, common_only: bool = True) -> np.ndarray:
    """Фотомозаика: каждая клетка — ближайший пластик, а ошибка цвета рассеивается на соседей
    (Флойд–Стайнберг в Lab). Градиенты и полутона живописи сохраняются как «зерно», как в
    LEGO Art; зато деталей больше (одноцветных полей почти нет)."""
    from .colors import _rgb_to_lab, studio_palette
    palette = studio_palette(common_only)
    pal_lab = _rgb_to_lab(np.array([c.rgb for c in palette]))
    pal_codes = np.array([c.code for c in palette])
    lab = _rgb_to_lab(colors.reshape(-1, 3)).reshape(colors.shape).astype(float)
    w, h = present.shape
    codes = np.full(present.shape, -1)
    for y in range(h):
        for x in range(w):
            if not present[x, y]:
                continue
            target = lab[x, y]
            i = int(((pal_lab - target) ** 2).sum(1).argmin())
            codes[x, y] = pal_codes[i]
            err = target - pal_lab[i]
            for dx, dy, share in ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)):
                if 0 <= x + dx < w and 0 <= y + dy < h and present[x + dx, y + dy]:
                    lab[x + dx, y + dy] += err * share
    return codes


def _vivid(cells: np.ndarray) -> np.ndarray:
    """Живопись и затенённые картинки: пластик насыщеннее краски. Насыщенность усиливается
    в VIVID_CHROMA раз — бордовые доспехи становятся красными, а не «коричневыми»."""
    from .colors import _rgb_to_lab
    lab = _rgb_to_lab(cells).astype(float)
    lab[:, 1:] *= VIVID_CHROMA                       # только насыщенность: светлота цветных не трогается,
                                                     # иначе тёмно-красное становится розовым
    return (np.clip(color.lab2rgb(lab.reshape(1, -1, 3)), 0, 1).reshape(-1, 3) * 255).astype(np.uint8)


VIVID_CHROMA = 1.25


def _grey_tones(cells: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Режим «контраст» для тёмных и блёклых картинок: бесцветные клетки раскладываются по четырём
    серым тонам пластика по светлоте, растянутой так, чтобы самое тёмное в картинке стало чёрным,
    а самое светлое — белым (2-й и 98-й перцентили). Тёмная картинка остаётся тёмной, но
    светотень внутри неё расходится по тонам, а не сваливается в один. Цветные клетки — как есть."""
    from .colors import _rgb_to_lab, code_by_name, studio_palette
    lab = _rgb_to_lab(cells)
    grey = np.hypot(lab[:, 1], lab[:, 2]) < 15
    out = codes.copy()
    if grey.sum() < 2:
        return out
    L = lab[grey, 0]
    lo, hi = np.quantile(L, 0.02), np.quantile(L, 0.98)
    stretched = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1) * 100
    tones = [code_by_name(n) for n in GREY_TONES]
    tone_L = np.array([_rgb_to_lab(np.array([[*c.rgb]], dtype=np.uint8))[0, 0] for c in studio_palette(False) if c.code in tones])
    tone_L = np.array([tone_L[[c.code for c in studio_palette(False) if c.code in tones].index(t)] for t in tones])
    edges = (tone_L[:-1] + tone_L[1:]) / 2
    out[grey] = np.array(tones)[np.searchsorted(edges, stretched)]
    return out


GREY_TONES = ("Black", "Dark_Bluish_Gray", "Light_Bluish_Gray", "White")


GREY_TONES = ("Black", "Dark_Bluish_Gray", "Light_Bluish_Gray", "White")
GREY_SHARES = (0.35, 0.35, 0.2, 0.1)   # доли бесцветных клеток по тонам, от тёмного к светлому


CONTRAST_MIX = 0.5   # доля выравнивания: 0 — как есть, 1 — полная эквализация


def write_check(result: PhotoMosaic, path: str) -> str:
    """Картинка «клетки фото | клетки LEGO» и сводка точности цвета: средняя ΔE и доля клеток
    с заметным (> NOTICEABLE_DE) отклонением. Это проверка результата, а не Studio под лампой."""
    from .colors import _rgb_to_lab
    m, cells = result.mosaic, result.cell_colors
    lego = {c.code: np.array(c.rgb, dtype=np.uint8) for c in studio_palette(common_only=False)}
    present = m.codes >= 0
    nx, ny = m.codes.shape
    photo_img = np.full((ny, nx, 3), 190, np.uint8)
    lego_img = photo_img.copy()
    for i, j in np.argwhere(present):
        photo_img[j, i] = cells[i, j]
        lego_img[j, i] = lego[int(m.codes[i, j])]
    scale = max(1, 600 // ny)
    a = Image.fromarray(photo_img).resize((nx * scale, ny * scale), Image.NEAREST)
    b = Image.fromarray(lego_img).resize((nx * scale, ny * scale), Image.NEAREST)
    out = Image.new("RGB", (a.width * 2 + scale, a.height), "white")
    out.paste(a, (0, 0)); out.paste(b, (a.width + scale, 0))
    out.save(path)
    codes = m.codes[present]
    de = np.sqrt(((_rgb_to_lab(cells[present]) - _rgb_to_lab(np.array([lego[int(c)] for c in codes]))) ** 2).sum(1))
    names = {c.code: c.name for c in studio_palette(common_only=False)}
    worst = sorted(((de[codes == c].mean(), int((codes == c).sum()), names[int(c)]) for c in np.unique(codes)), reverse=True)[:3]
    return (f"точность цвета: средняя ΔE {de.mean():.1f}, клеток с заметным отклонением (ΔE > {NOTICEABLE_DE}) "
            f"{int((de > NOTICEABLE_DE).sum())} из {len(de)} ({(de > NOTICEABLE_DE).mean() * 100:.0f}%); "
            "дальше всего от фото: " + ", ".join(f"{n} (ΔE {d:.0f}, {k} кл.)" for d, k, n in worst))


NOTICEABLE_DE = 20  # ΔE, с которого разница цветов бросается в глаза


def _cutout(path):
    """(rgb 0..1, маска объекта). Рисунок на ровном фоне (пиксель-арт, скриншот): фон — цвет
    рамки, маска — всё, что от него отличается, и линии сетки строго горизонтальны/вертикальны;
    rembg тут только вредит (съедает тонкие линии). Иначе — фото, фон вырезает rembg
    (у товарного фото на белом рамка тоже ровная, но швы идут под углом перспективы)."""
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image).astype(float) / 255
    inset = max(2, min(rgb.shape[:2]) // 100)                 # у самого края бывает тёмная кромка
    border = np.concatenate([rgb[inset], rgb[-1 - inset], rgb[:, inset], rgb[:, -1 - inset]])
    if border.std(axis=0).max() < FLAT_BORDER_STD:
        background = np.median(border, axis=0)
        flat_mask = np.abs(rgb - background).max(axis=2) > FLAT_BACKGROUND_TOLERANCE
        if 0.02 < flat_mask.mean() < 0.9 and is_flat(rgb, flat_mask):
            return rgb, flat_mask
    import rembg
    rgba = np.asarray(rembg.remove(image))
    return rgba[..., :3].astype(float) / 255, rgba[..., 3] > 128


def is_flat(rgb, mask) -> bool:
    """Плоская картинка: линии сетки строго горизонтальны и вертикальны — выправлять нечего."""
    try:
        segs, angle = _segments(rgb, mask)
    except ValueError:
        return False
    axis_aligned = (np.minimum(angle, 180 - angle) < 1.5) | (np.abs(angle - 90) < 1.5)
    return axis_aligned.mean() > 0.9


def _segments(rgb, mask):
    """Отрезки линий (Хаф) на швах сетки и их углы в градусах (0..180)."""
    gray = color.rgb2gray(rgb)
    edges = feature.canny(gray, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    edges &= morphology.erosion(mask, morphology.disk(3))
    segs = np.array(transform.probabilistic_hough_line(edges, threshold=8, line_length=25, line_gap=2, rng=0), dtype=float)
    if len(segs) == 0:
        raise ValueError("не нашёл линии сетки на фото")
    angle = np.degrees(np.arctan2(segs[:, 1, 1] - segs[:, 0, 1], segs[:, 1, 0] - segs[:, 0, 0])) % 180
    return segs, angle


def _rectify(rgb, mask):
    """Гомография по точкам схода двух семейств линий сетки; отражение убирается.
    Плоская картинка возвращается как есть — любое выправление только сдвинет её сетку."""
    if is_flat(rgb, mask):
        return rgb, mask
    segs, angle = _segments(rgb, mask)
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
    J = _jacobian(H, center)                            # гомография может отразить любую из осей
    sx = -scale if J[0, 0] < 0 else scale
    sy = -scale if J[1, 1] < 0 else scale
    fit = np.array([[sx, 0, (hi[0] if sx < 0 else -lo[0]) * scale + 20],
                    [0, sy, (hi[1] if sy < 0 else -lo[1]) * scale + 20], [0, 0, 1]])
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
        # основной период — наименьший из пиков, почти равных лучшему: у чистого пиксель-арта
        # пики на 2p, 3p не ниже пика на p
        top = max(peaks)[0]
        l = min(l for v, l in peaks if v >= top * FUNDAMENTAL_RATIO)
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


def _anchors(colors, present, front):
    """Опорные точки уровней: чёрный — пластик боковой грани (если виден), белый — самые
    светлые клетки спрайта (если они близки к белому)."""
    rgb = colors / 255.0
    gray = color.rgb2gray(rgb)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    side = present & ~front
    # чёрный — самое тёмное из срезанного (пластик), а не среднее: в срез могут попасть и клетки контура
    black = np.median(colors[side][gray[side] <= np.quantile(gray[side], 0.3)], axis=0) if side.sum() >= 3 else None
    # белый — только бесцветные светлые клетки; кремовое лицо за белый не сойдёт
    whitish = front & (gray > 0.85) & (chroma < 0.12)
    white = np.median(colors[whitish], axis=0) if whitish.sum() >= 3 else None
    return black, white


def _front_face(colors, present):
    """Передняя грань: силуэт, «сжатый» на вектор ребра (см. описание модуля), одним связным куском."""
    from scipy import ndimage
    rgb = colors / 255.0
    dark = present & (color.rgb2gray(rgb) < DARK_L) & ((rgb.max(axis=2) - rgb.min(axis=2)) < DARK_CHROMA)
    best, best_score = present, 0
    for ex in range(-MAX_SIDE, MAX_SIDE + 1):
        for ey in range(-MAX_SIDE, MAX_SIDE + 1):
            if (ex, ey) == (0, 0):
                continue
            front = present.copy()
            steps = max(abs(ex), abs(ey))
            for t in range(1, steps + 1):
                front &= _shift(present, round(ex * t / steps), round(ey * t / steps))
            removed = present & ~front
            edge = front & ndimage.binary_dilation(removed)          # клетки, ставшие краем
            score = int((removed & dark).sum()) - 3 * int((removed & ~dark).sum()) - 2 * int((edge & ~dark).sum())
            if score > best_score:
                best, best_score = front, score
    best = _drop_base(best, dark)
    labels, n = ndimage.label(best)
    if n > 1:
        sizes = ndimage.sum(best, labels, range(1, n + 1))
        best = labels == (int(np.argmax(sizes)) + 1)
    return best


def _drop_base(front, dark):
    """Подставка: нижние ряды, целиком тёмные и сплошные (без просвета между ногами) на большую
    часть ширины фигурки. Контур под ботинками — не подставка: у него есть разрыв."""
    front = front.copy()
    width = int(np.diff(np.nonzero(front.any(axis=1))[0][[0, -1]])[0]) + 1 if front.any() else 0
    for y in range(front.shape[1] - 1, -1, -1):
        row = front[:, y]
        if not row.any():
            continue
        xs = np.nonzero(row)[0]
        solid = xs[-1] - xs[0] + 1 == len(xs)                     # без разрывов
        if row.sum() == (row & dark[:, y]).sum() and solid and len(xs) >= BASE_MIN_WIDTH * width:
            front[:, y] = False
        else:
            break
    return front


def _shift(a, dx, dy):
    """a, сдвинутый так, что out[c] = a[c + (dx, dy)] (за краем — False)."""
    out = np.zeros_like(a)
    xs = slice(max(0, dx), a.shape[0] + min(0, dx)); ys = slice(max(0, dy), a.shape[1] + min(0, dy))
    xd = slice(max(0, -dx), a.shape[0] + min(0, -dx)); yd = slice(max(0, -dy), a.shape[1] + min(0, -dy))
    out[xd, yd] = a[xs, ys]
    return out
