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
from scipy import ndimage
from skimage import color, feature, morphology, transform

from .colors import DARK_L_FLAT, flat_codes, studio_palette
from .colors import DARK_L as DARK_L_PHOTO
from .mosaic import Mosaic
from .preferences import preferences

RECTIFIED_PX = 900          # длинная сторона выправленного изображения
MIN_PITCH, MAX_PITCH = 8, 80
FUNDAMENTAL_RATIO = 0.9           # решётка собирает не меньше 90 % границ от лучшей — тот же период, берём меньший шаг
EDGE_MIN = 0.2                    # сильная граница — не ниже 20 % от максимума профиля
FLAT_BORDER_STD = 0.03            # рамка картинки одноцветная — это рисунок на ровном фоне, не фото
FLAT_BACKGROUND_TOLERANCE = 0.12  # насколько цвет должен отличаться от фона, чтобы быть объектом
ALPHA_BACKGROUND = 0.02           # доля прозрачных пикселей, с которой фон считается прозрачным
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
    rgb, mask, known_flat = _cutout(image_path, keep_background)
    rect, rmask, flat = _rectify(rgb, mask, known_flat)
    bounds_x, bounds_y = _grid(rect, rmask, flat)
    colors, present = _sample_cells(rect, rmask, bounds_x, bounds_y, flat)
    if keep_background:
        colors, present = _fill_panel(colors, _trim_panel(present))
    colors, present = _symmetrize(colors, present)
    front = present if keep_background else _front_face(colors, present)
    black, white = _anchors(colors, present, front)
    codes = np.full(present.shape, -1)
    prefs = preferences()["mosaic"]
    codes[front], calibrated = flat_codes(colors[front], max_colors, black, white,
                                          outline_black=prefs["outline_black"], common_only=prefs["palette"] == "common",
                                          dark_l=DARK_L_FLAT if flat else DARK_L_PHOTO, exact_hues=flat)
    cell_colors = np.zeros_like(colors)
    cell_colors[front] = calibrated
    pitch = (np.diff(bounds_x).mean() + np.diff(bounds_y).mean()) / 2
    return PhotoMosaic(Mosaic(codes, pitch, *present.shape), cell_colors)


def mosaic_from_image(image_path: str, width: int, max_colors: int = 12, keep_background: bool = False,
                      outline: bool = False, contrast: bool = False, dither: bool = False) -> PhotoMosaic:
    """Любая картинка → пиксель-арт шириной `width` клеток. Объект (или вся картинка при
    keep_background) режется на клетки, цвет клетки — медиана её пикселей, потом ближайший
    пластик. Одиночные клетки без соседей убираются; outline — чёрный контур в одну клетку
    вокруг силуэта, как у Pixel Pals."""
    from scipy import ndimage
    rgb, mask, _ = _cutout(image_path)
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


def _cutout(path, keep_background: bool = False):
    """(rgb 0..1, маска объекта, плоская ли картинка — True, если это известно наверняка).
    keep_background — панно: маска — вся картинка, фон не ищем (rembg не нужен).
    PNG с прозрачным фоном: маска — альфа, картинка заведомо плоская (у фото альфы не бывает; в RGB
    прозрачное — чёрное, и чёрный контур пропал бы как фон). Рисунок на ровном фоне (пиксель-арт,
    скриншот): фон — цвет рамки, маска — всё, что от него отличается, и линии сетки строго
    горизонтальны/вертикальны; rembg тут только вредит (съедает тонкие линии). Иначе — фото, фон
    вырезает rembg (у товарного фото на белом рамка тоже ровная, но швы идут под углом перспективы)."""
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    image = Image.fromarray(rgba[..., :3])
    rgb = rgba[..., :3].astype(float) / 255
    alpha_mask = rgba[..., 3] >= 128
    if keep_background or is_full_frame(rgb):
        return rgb, np.ones(alpha_mask.shape, dtype=bool), False
    if (~alpha_mask).mean() > ALPHA_BACKGROUND:
        return rgb, alpha_mask, True
    inset = max(2, min(rgb.shape[:2]) // 100)                 # у самого края бывает тёмная кромка
    border = np.concatenate([rgb[inset], rgb[-1 - inset], rgb[:, inset], rgb[:, -1 - inset]])
    if border.std(axis=0).max() < FLAT_BORDER_STD:
        background = np.median(border, axis=0)
        flat_mask = np.abs(rgb - background).max(axis=2) > FLAT_BACKGROUND_TOLERANCE
        if 0.02 < flat_mask.mean() < 0.9 and is_flat(rgb, flat_mask):
            return rgb, _fill_hollow(flat_mask), True
    mask = _fill_small_holes(_object_mask(rgb, image))
    # Вне маски — чёрное (как в выводе rembg), а не фон: при выправлении перспективы и в краевых
    # клетках примешивается именно оно, и краевые клетки темнеют — как боковые грани, которые
    # и отбрасываются. Цвет самой фигуры — из оригинала: под залитыми дырами у rembg тоже чёрное.
    return np.where(mask[..., None], rgb, 0.0), mask, False


def is_full_frame(rgb) -> bool:
    """Картинка занимает весь кадр — фото панно, скан картины: по рамке идут разные цвета, а не фон.
    У предмета на фоне (хоть на фактурном) рамка одного цвета: разброс там 0–2, у панно — за 30."""
    lab = color.rgb2lab(rgb)
    inset = max(2, min(lab.shape[:2]) // 100)
    border = np.concatenate([lab[inset], lab[-1 - inset], lab[:, inset], lab[:, -1 - inset]])
    spread = np.sqrt(((border - np.median(border, axis=0)) ** 2).sum(1))
    return float(np.median(spread)) > BORDER_SPREAD


BORDER_SPREAD = 10.0   # ΔE по рамке, выше которого это не фон, а сама картинка


def _object_mask(rgb, image) -> np.ndarray:
    """Маска предмета на непростом фоне. rembg (нейросеть) точнее, но в браузере её нет — там
    заливка от рамки по близкому цвету: на проверенных фото она совпадает с rembg на 94–99 %."""
    try:
        import onnxruntime
        import rembg
    except ImportError:
        return _flood_object(rgb)
    onnxruntime.disable_telemetry_events()   # иначе onnxruntime падает (abort) при выходе из Python — поток телеметрии
    return np.asarray(rembg.remove(image))[..., 3] > 128


def _flood_object(rgb) -> np.ndarray:
    """Заливка от рамки: фон — связные куски цвета рамки, касающиеся края; остальное — предмет."""
    lab = color.rgb2lab(rgb)
    inset = max(2, min(lab.shape[:2]) // 100)
    border = np.concatenate([lab[inset], lab[-1 - inset], lab[:, inset], lab[:, -1 - inset]])
    close = np.sqrt(((lab - np.median(border, axis=0)) ** 2).sum(2)) < FLOOD_TOLERANCE
    labels, count = ndimage.label(close)
    edge = {int(l) for l in np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]) if l}
    background = np.isin(labels, sorted(edge)) if edge else np.zeros(labels.shape, bool)
    return ndimage.binary_fill_holes(~background)


FLOOD_TOLERANCE = 18.0   # ΔE: насколько пиксель может отличаться от цвета рамки и всё ещё считаться фоном


def _fill_hollow(mask):
    """Белая собака на белом: тело цвета фона внутри контура — в маске одна обводка. Если дыр
    внутри главного контура больше HOLLOW его площади, это полая фигура, и дыры — её тело.
    У настоящих просветов (между рукой и телом) дыр ≤ 0.2 площади."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    main = labels == (np.argmax(ndimage.sum(mask, labels, range(1, n + 1))) + 1)
    holes = ndimage.binary_fill_holes(main) & ~main
    return mask | holes if holes.sum() > HOLLOW * main.sum() else mask


HOLLOW = 0.5


def _fill_small_holes(mask):
    """rembg считает мелкие детали внутри фигуры фоном (жёлтый глаз на бежевом фоне, тёмная точка):
    дыры меньше SMALL_HOLE площади фигуры заливаем, большие просветы (между крылом и хвостом) — нет."""
    holes, n = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)
    if not n:
        return mask
    small = ndimage.sum(np.ones_like(mask), holes, range(1, n + 1)) < mask.sum() * SMALL_HOLE
    return mask | np.isin(holes, np.flatnonzero(small) + 1)


SMALL_HOLE = 0.01


def is_flat(rgb, mask) -> bool:
    """Плоская картинка: линии сетки строго горизонтальны и вертикальны — выправлять нечего."""
    try:
        segs, angle = _segments(rgb, mask)
    except ValueError:
        return False
    axis_aligned = (np.minimum(angle, 180 - angle) < FLAT_ANGLE) | (np.abs(angle - 90) < FLAT_ANGLE)
    return axis_aligned.mean() > 0.9


FLAT_ANGLE = 1.5   # градусов: у ровного снимка мозаики короткие отрезки по краям плиток дают ±3°, у снимка
                   # под углом (перспектива) линии расходятся сильнее — доля осевых падает до ~0.5


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


def _rectify(rgb, mask, known_flat: bool = False):
    """Гомография по точкам схода двух семейств линий сетки; отражение убирается.
    Плоская картинка возвращается как есть — любое выправление только сдвинет её сетку."""
    if known_flat or is_flat(rgb, mask):
        return rgb, mask, True
    segs, angle = _segments(rgb, mask)
    horizontal = (angle < 20) | (angle > 175)
    vertical = (angle > 70) & (angle < 110)
    if horizontal.sum() < 5 or vertical.sum() < 5:
        raise ValueError("не нашёл линии сетки на фото")
    center = np.array([rgb.shape[1] / 2, rgb.shape[0] / 2, 1.0])
    vps = [_vanishing_point(segs[horizontal]), _vanishing_point(segs[vertical])]
    if all(_vanishing_distance(vp, rgb.shape) > NO_PERSPECTIVE for vp in vps):
        H = np.eye(3)   # линии параллельны — перспективы нет; гомография по далёким точкам схода только исказила бы сетку
    else:
        H = np.linalg.inv(np.c_[vps[0], vps[1], center])
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
    return rect, rmask, False


NO_PERSPECTIVE = 50   # точки схода дальше стольких размеров кадра — снимок ровный (у снимка под углом: 7–31)


def _vanishing_distance(vp, shape) -> float:
    """Расстояние точки схода от центра кадра в размерах кадра; бесконечность — в бесконечности."""
    if abs(vp[2]) < 1e-12:
        return np.inf
    return float(np.hypot(vp[0] / vp[2] - shape[1] / 2, vp[1] / vp[2] - shape[0] / 2) / max(shape[:2]))


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


def _grid(rect, rmask, flat: bool):
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
        # Кандидаты — пики автокорреляции. Выбор — по решётке: доля сильных границ профиля,
        # попавших (±1.5 px) на узлы решётки с шагом-кандидатом при лучшей фазе. Настоящий шаг
        # собирает все границы; вдвое крупный — половину; вдвое мелкий промахивается на нечётных
        # узлах. Сам пик автокорреляции врёт: у спрайтов с двухпиксельными деталями пик на 2p выше.
        top = max(v for v, _ in peaks)
        edges = [i for i in range(1, len(profile) - 1)
                 if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > profile.max() * EDGE_MIN]
        options = []
        for v, l in peaks:
            if v < top * 0.3:
                continue
            a, b, c = ac[l - 1], ac[l], ac[l + 1]
            pitch = l + 0.5 * (a - c) / (a - 2 * b + c)              # субпиксельно, по параболе
            fit = lambda ph: np.mean([min((e - ph) % pitch, pitch - (e - ph) % pitch) <= 1.5 for e in edges]) if edges else 0.0
            phase = max(range(int(pitch)), key=fit)
            options.append((fit(phase), pitch, phase))
        best = max(o[0] for o in options)
        _, pitch, phase = min((o for o in options if o[0] >= best * FUNDAMENTAL_RATIO), key=lambda o: o[1])
        # прослеживание — только у плоского рисунка: у фото границы размыты, не по чему идти
        out.append(_track_boundaries(pitch, phase, np.array(edges, float) if flat else np.array([]), len(profile) + 1))
    return out   # границы клеток по x, по y


def _track_boundaries(pitch, phase, edges, length):
    """Границы клеток вдоль оси: от фазы шагаем на шаг, и если рядом (±SNAP шага) есть граница
    пикселей рисунка — встаём на неё. Ровная решётка «фаза + k·шаг» на реальных картинках не
    держится: у отмасштабированного с нецелым коэффициентом спрайта клетки по 11 и 12 px
    чередуются неравномерно, и к дальнему краю решётка уезжает на полклетки — клетки садятся
    между пикселями рисунка, цвета смешиваются. Прослеживание держит ошибку в пределах клетки."""
    bounds = [phase]
    while bounds[-1] + pitch <= length:
        expected = bounds[-1] + pitch
        near = edges[np.abs(edges - expected) <= pitch * SNAP]
        bounds.append(float(near[np.abs(near - expected).argmin()]) if len(near) else expected)
    return np.array(bounds)


SNAP = 0.3   # доля шага, в пределах которой граница клетки притягивается к границе пикселей


def _sample_cells(rect, rmask, bounds_x, bounds_y, flat: bool):
    """Цвет — медиана внутренней части клетки; клетка есть, если маска покрывает её большую часть.
    У плоского рисунка цвет краевой клетки берётся только из глубины маски: по кромке лежит
    полупрозрачная кайма от масштабирования, и медиана по всей клетке даёт серый. У фото —
    по всей клетке: вне маски чёрное, и клетка, срезанная маской, темнеет — это боковая грань
    или перспектива, и такие клетки отбрасывает _front_face."""
    nx, ny = len(bounds_x) - 1, len(bounds_y) - 1
    pitch = min(np.diff(bounds_x).mean(), np.diff(bounds_y).mean())
    core = morphology.binary_erosion(rmask, morphology.disk(max(1, int(pitch * FRINGE)))) if flat else rmask
    colors = np.zeros((nx, ny, 3), dtype=np.uint8)
    present = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            (x0, x1), (y0, y1) = bounds_x[i:i + 2], bounds_y[j:j + 2]
            xs = slice(int(x0 + (x1 - x0) * 0.3), int(x0 + (x1 - x0) * 0.7))
            ys = slice(int(y0 + (y1 - y0) * 0.3), int(y0 + (y1 - y0) * 0.7))
            if rmask[ys, xs].mean() < 0.6:
                continue
            present[i, j] = True
            pixels = rect[ys, xs][core[ys, xs]] if flat else rect[ys, xs].reshape(-1, 3)
            if len(pixels) < MIN_CORE_PIXELS:
                pixels = rect[ys, xs][rmask[ys, xs]]
            colors[i, j] = (np.median(pixels, axis=0) * 255).astype(np.uint8)
    return colors, present


FRINGE = 0.2           # ширина каймы у края силуэта в долях шага сетки — эти пиксели в цвет не идут
MIN_CORE_PIXELS = 4    # меньше — клетка целиком кайма, берём медиану по всем пикселям маски


def _trim_panel(present):
    """Панно — полный прямоугольник: крайние ряды и столбцы, где клеток не хватает (край фото
    срезал мозаику, выправленный кадр вышел за снимок), отбрасываются целиком."""
    present = present.copy()
    while present.any():
        ys, xs = np.nonzero(present.any(axis=0))[0], np.nonzero(present.any(axis=1))[0]
        rows = present.sum(axis=0) / len(xs)      # доля клеток в ряду от ширины оставшегося панно
        cols = present.sum(axis=1) / len(ys)
        edges = [("y", ys[0]), ("y", ys[-1]), ("x", xs[0]), ("x", xs[-1])]
        worst = min(edges, key=lambda e: rows[e[1]] if e[0] == "y" else cols[e[1]])
        fill = rows[worst[1]] if worst[0] == "y" else cols[worst[1]]
        if fill >= PANEL_FULL:
            break
        if worst[0] == "y":
            present[:, worst[1]] = False
        else:
            present[worst[1], :] = False
    return present


PANEL_FULL = 0.95   # ряд панно считается целым, если клеток не меньше этой доли


def _fill_panel(colors, present):
    """Одиночные пропуски внутри панно (клетка на самом краю кадра) — цвет соседей."""
    ys, xs = np.nonzero(present.any(axis=0))[0], np.nonzero(present.any(axis=1))[0]
    colors, present = colors.copy(), present.copy()
    for x in range(xs.min(), xs.max() + 1):
        for y in range(ys.min(), ys.max() + 1):
            if present[x, y]:
                continue
            around = [colors[i, j] for i, j in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                      if 0 <= i < present.shape[0] and 0 <= j < present.shape[1] and present[i, j]]
            if around:
                colors[x, y] = np.median(np.array(around), axis=0).astype(np.uint8)
                present[x, y] = True
    return colors, present


def _symmetrize(colors, present):
    """Симметричная фигура — симметричная мозаика. У «векторного» пиксель-арта блоки нарисованы
    вручную и разной ширины (22–32 px), регулярной сетки нет, и клетки ошибаются то влево, то
    вправо — левая и правая щека выходят разными. Если силуэт и цвета почти совпадают с зеркалом,
    зеркальные пары клеток получают один цвет (среднее) и одно присутствие."""
    xs = np.nonzero(present.any(axis=1))[0]
    if len(xs) == 0:
        return colors, present
    lo, hi = xs.min(), xs.max()
    sub, sub_colors = present[lo:hi + 1], colors[lo:hi + 1].astype(float)
    mirror, mirror_colors = sub[::-1], sub_colors[::-1]
    both = sub & mirror
    iou = both.sum() / (sub | mirror).sum()
    differ = (np.abs(sub_colors - mirror_colors).max(axis=2)[both] > SYMMETRY_COLOR * 255).mean()
    if iou < SYMMETRY_IOU or differ > SYMMETRY_DIFFER:
        return colors, present
    colors, present = colors.copy(), present.copy()
    present[lo:hi + 1] = both
    colors[lo:hi + 1] = ((sub_colors + mirror_colors) / 2).astype(np.uint8)
    return colors, present


SYMMETRY_IOU, SYMMETRY_COLOR, SYMMETRY_DIFFER = 0.88, 0.15, 0.12   # у несимметричных фигур цвета расходятся ≥ 30 % клеток


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
