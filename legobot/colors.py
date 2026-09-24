"""Палитра цветов LDraw (catalog/colors.csv, собирается tools/catalog.py из Studio) и подбор цветов модели.

Цвета сравниваются в CIELAB — расстояние там соответствует тому, как видит глаз.
Цвета модели сначала сводятся к нескольким доминирующим (k-means): у модели
из деталей цветов мало, а тени и блики с фото — не цвета.
"""
import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import catalog

import numpy as np
from scipy.cluster.vq import kmeans2

COLORS_CSV = Path(__file__).resolve().parent.parent / "catalog" / "colors.csv"

# Материалы, которые не годятся для обычных деталей: прозрачные, металлики, резина и т.п.
_SPECIAL = ("ALPHA", "CHROME", "PEARLESCENT", "RUBBER", "MATTE_METALLIC", "METAL", "MATERIAL", "LUMINANCE")

# Ходовые сплошные цвета: то, что реально есть в продаже в любых деталях.
COMMON_COLORS = {
    0, 1, 2, 4, 14, 15, 19, 25, 27, 28, 70, 71, 72, 73, 74, 84, 85, 191, 212, 226, 272, 288, 308, 320, 321, 322, 323, 326, 378, 379, 462, 484,
    5, 13, 22, 26, 29, 30, 31, 69, 112,   # розовые, пурпурные, лиловые
    78, 86, 92,                           # телесные: Light_Flesh, Dark_Flesh, Flesh
    151,                                  # Very_Light_Bluish_Gray — светло-серый фон панно
    330,                                  # Olive_Green — хаки, ближе к нему у зелёно-жёлтого ничего нет
}

_LINE = re.compile(r"^0 !COLOUR (\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})")


@dataclass(frozen=True)
class LdrawColor:
    code: int
    name: str
    rgb: tuple[int, int, int]


@lru_cache
def _rows() -> tuple[dict, ...]:
    with open(COLORS_CSV, newline="") as f:
        return tuple(csv.DictReader(f))


def _color(row, studio: bool = False) -> LdrawColor:
    hexrgb = row["studio_rgb"] if studio else row["rgb"]
    return LdrawColor(int(row["code"]), row["name"], tuple(int(hexrgb[i:i + 2], 16) for i in (0, 2, 4)))


@lru_cache
def load_palette(common_only: bool = True) -> tuple[LdrawColor, ...]:
    """Сплошные цвета в порядке LDConfig; common_only — только ходовые."""
    return tuple(_color(r) for r in _rows() if r["solid"] == "1" and (not common_only or r["common"] == "1"))


@lru_cache
def all_colors() -> tuple[LdrawColor, ...]:
    """Каждый цвет LDraw, включая прозрачные и металлики: в чужой модели встречается любой."""
    return tuple(_color(r) for r in _rows())


@lru_cache
def all_color_names() -> dict[int, str]:
    """Все цвета, включая прозрачные и металлики — для чтения чужих моделей."""
    return {int(r["code"]): r["name"] for r in _rows()}


@lru_cache
def studio_palette(common_only: bool = True) -> tuple[LdrawColor, ...]:
    """Те же цвета с RGB из таблицы Studio — тем, как Studio их рисует. У справочника LDraw
    часть значений расходится (Light_Purple: #cd6298 против #af3195 в Studio)."""
    return tuple(_color(r, studio=True) for r in _rows() if r["solid"] == "1" and (not common_only or r["common"] == "1"))


LIGHTNESS_WEIGHT = 0.3  # при кластеризации яркость важна меньше оттенка: тень на жёлтом — всё ещё жёлтый


def nearest_codes(rgb: np.ndarray, max_colors: int = 4) -> np.ndarray:
    """rgb: uint8 [N, 3] -> коды LDraw [N]. Сначала k-means до max_colors доминирующих цветов
    (в Lab с приглушённой яркостью, чтобы тени не становились отдельными цветами),
    затем центр каждого кластера — к ближайшему цвету палитры."""
    rgb = rgb.reshape(-1, 3)
    if len(rgb) == 0:
        return np.zeros(0, dtype=int)
    lab = _rgb_to_lab(rgb)
    weighted = lab * np.array([LIGHTNESS_WEIGHT, 1.0, 1.0])
    k = min(max_colors, len(np.unique(rgb, axis=0)))
    _, labels = kmeans2(weighted, k, minit="++", seed=0)
    labels = _merge_same_hue(lab, labels, k)
    palette = load_palette()
    palette_lab = _rgb_to_lab(np.array([c.rgb for c in palette]))
    codes = np.array([c.code for c in palette])
    center_codes = np.zeros(k, dtype=int)
    for i in range(k):
        members = lab[labels == i] if (labels == i).any() else lab
        center_codes[i] = _match_cluster(members, palette_lab, codes)
    return center_codes[labels]


def panel_codes(colors: np.ndarray, present: np.ndarray, black: np.ndarray | None = None,
                white: np.ndarray | None = None, common_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Подбор для панно — фотографии выложенной картины.

    У картины нет ни контура, ни плоских заливок: полутона и тени — её часть. Поэтому эвристики
    фигурки картину портят: «всё тёмное и неяркое — чёрный контур» съедает ночное небо и крыши,
    а k-means на десяток-другой цветов уходит целиком в самую большую область, и мелким местам
    цветов не остаётся. Здесь клетка берёт ближайший продающийся пластик, но с оглядкой на
    соседей: за каждого соседа другого цвета к расстоянию добавляется SMOOTH_DE. Клетка, что
    болталась между двумя почти одинаковыми синими, садится на цвет соседей, а настоящая деталь
    (жёлтое окно среди синего) эту доплату перевешивает и остаётся.

    Возвращает (коды [W, H], -1 где пусто; откалиброванные цвета клеток present)."""
    lo = np.zeros(3) if black is None else np.asarray(black, float)
    hi = np.full(3, 255.0) if white is None else np.asarray(white, float)
    calibrated = np.clip((colors[present].astype(float) - lo) / np.maximum(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    palette = studio_palette(common_only)
    if common_only and catalog.purchasable_colors() is not None:
        palette = tuple(c for c in palette if c.code in catalog.purchasable_colors())
    palette_lab = _rgb_to_lab(np.array([c.rgb for c in palette]))
    distance = np.full(present.shape + (len(palette),), np.inf)
    distance[present] = np.sqrt(((_rgb_to_lab(calibrated)[:, None] - palette_lab[None]) ** 2).sum(2))
    chosen = _smooth_choice(distance, present)
    codes = np.full(present.shape, -1)
    codes[present] = np.array([c.code for c in palette])[chosen[present]]
    return codes, calibrated


def _smooth_choice(distance: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Цвет каждой клетки: расстояние до пластика плюс SMOOTH_DE за каждого соседа другого цвета.
    Проходов несколько — каждый считает соседей по выбору предыдущего."""
    chosen = np.argmin(distance, axis=2)
    for _ in range(SMOOTH_PASSES):
        same, neighbours = np.zeros(distance.shape), np.zeros(present.shape)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = _shift(np.where(present, chosen, -1), dx, dy)
            near = shifted >= 0
            neighbours += near
            same[near, shifted[near]] += 1
        chosen = np.argmin(distance + SMOOTH_DE * (neighbours[..., None] - same), axis=2)
    return chosen


def _shift(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Сетка, сдвинутая на (dx, dy); за краем -1."""
    out = np.full_like(grid, -1)
    w, h = grid.shape
    out[max(-dx, 0):w + min(-dx, 0), max(-dy, 0):h + min(-dy, 0)] = grid[max(dx, 0):w + min(dx, 0), max(dy, 0):h + min(dy, 0)]
    return out


SMOOTH_DE = 4.0      # ΔE за соседа другого цвета: столько стоит одинокая крапина в картине
SMOOTH_PASSES = 3    # дальше выбор почти не меняется


def flat_codes(rgb: np.ndarray, max_colors: int, black: np.ndarray | None = None,
               white: np.ndarray | None = None, outline_black: bool = True,
               common_only: bool = True, dark_l: float | None = None, exact_hues: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Подбор для пиксель-арта: цвета плоские, теней нет. Сначала уровни: что на фото было
    `black`/`white` (чёрный пластик, белые клетки), становится чёрным/белым — фото бледнее
    пластика. Потом k-means в Lab, слияние кластеров, разбитых освещением, и ближайший ходовой
    цвет в значениях Studio; у цветных — только среди своего оттенка (розовый не станет лиловым).
    exact_hues — цвета точные (цифровой рисунок, не фото): насыщенным не даётся фора базовым
    цветам, лазурь остаётся Dark_Azure, а не Medium_Blue. У фото оттенки врут (синий пластик под
    лампой — фиолетовый), там фора нужна всегда.
    Возвращает (коды, откалиброванные цвета) — вторые нужны для проверки результата."""
    rgb = rgb.reshape(-1, 3).astype(float)
    lo = np.zeros(3) if black is None else np.asarray(black, float)
    hi = np.full(3, 255.0) if white is None else np.asarray(white, float)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    lab = _rgb_to_lab(rgb)
    k = min(max_colors, len(np.unique(rgb, axis=0)))
    centers, labels = kmeans2(lab, k, minit="++", seed=0)
    centers, labels = _merge_close(centers, labels)
    palette = studio_palette(common_only)
    if common_only and catalog.purchasable_colors() is not None:
        palette = tuple(c for c in palette if c.code in catalog.purchasable_colors())   # Pick a Brick: только то, что продаётся
    palette_lab = _rgb_to_lab(np.array([c.rgb for c in palette]))
    palette_hsv_hue = _hsv_hue(np.array([c.rgb for c in palette], dtype=float))
    codes = np.array([c.code for c in palette])
    basic = np.array([c.name in BASIC_COLORS for c in palette])
    center_hsv_hue = _hsv_hue(_lab_to_rgb(centers))
    ranked: list[np.ndarray | None] = []   # для каждого кластера — расстояния до пластика (inf — нельзя), None — чёрный
    for c, c_hue in zip(centers, center_hsv_hue):
        if outline_black and _is_black(c, DARK_L if dark_l is None else dark_l):
            ranked.append(None)          # тёмное и бесцветное — контур, чёрный пластик, тени: всё чёрным
            continue
        dist = np.sqrt(((c - palette_lab) ** 2).sum(1))
        if not exact_hues or np.hypot(c[1], c[2]) < DULL_CHROMA:
            dist[basic] -= BASIC_BONUS   # фигурки красят базовыми цветами, а не «тёмно-лиловым»
        pal_chroma = np.hypot(palette_lab[:, 1], palette_lab[:, 2])
        if np.hypot(c[1], c[2]) <= ACHROMATIC:
            dist[pal_chroma > GREY_CHROMA] = np.inf      # серое остаётся серым, а не «светло-бирюзовым»
        else:
            hue = np.degrees(np.arctan2(c[2], c[1]))
            pal_hue = np.degrees(np.arctan2(palette_lab[:, 2], palette_lab[:, 1]))
            off_hue = np.abs((pal_hue - hue + 180) % 360 - 180) > MAX_HUE_DIFF
            if np.hypot(c[1], c[2]) > DULL_CHROMA:
                off_hue |= pal_chroma < ACHROMATIC   # насыщенный цвет не станет серым; тусклый — может (Sand_Green)
                # Lab у тёмных насыщенных синих «съезжает» к фиолетовому: чистый синий (4, 45, 153) выходит
                # ближе к Dark_Purple, чем к Blue. Оттенок HSV этого не делает — штраф за его расхождение.
                dist = dist + HSV_HUE_WEIGHT * np.abs((palette_hsv_hue - c_hue + 180) % 360 - 180)
            if not off_hue.all():
                dist[off_hue] = np.inf
        ranked.append(dist)
    center_codes = _assign_distinct(centers, labels, ranked, codes, palette_lab)
    return np.array(center_codes)[labels], rgb


def _assign_distinct(centers, labels, ranked, codes, palette_lab) -> list[int]:
    """Каждому кластеру — ближайший пластик, но мелкая деталь другого цвета не сливается с
    крупной областью: глаза не должны исчезнуть в плаще, когда их неоновый цвет не продаётся. Кластеры
    идут от большого к малому; если ближайший пластик уже занят кластером другого цвета
    (ΔE между центрами > DISTINCT_DE), берётся следующий свободный, если он не дальше
    ближайшего более чем на COLLISION_SLACK; среди запасных предпочтителен близкий по оттенку
    (неоново-лиловые глаза — Medium_Lavender, а не Blue)."""
    order = sorted(range(len(centers)), key=lambda i: -int((labels == i).sum()))
    pal_hue = np.degrees(np.arctan2(palette_lab[:, 2], palette_lab[:, 1]))
    taken: dict[int, np.ndarray] = {}   # код пластика -> центр кластера, который его занял
    result = [BLACK] * len(centers)
    for i in order:
        if ranked[i] is None:
            continue
        dist = ranked[i]
        best = int(np.argmin(dist))
        choice = best

        def free(j) -> bool:
            owner = taken.get(int(codes[j]))
            return owner is None or np.sqrt(((owner - centers[i]) ** 2).sum()) <= DISTINCT_DE

        small = (labels == i).sum() <= SMALL_FEATURE * labels.size
        if small and not free(best):   # мелкая деталь, ближайший занят другим цветом — запасной, близкий и по ΔE, и по оттенку
            hue = np.degrees(np.arctan2(centers[i][2], centers[i][1]))
            hue_penalty = HUE_WEIGHT * np.abs((pal_hue - hue + 180) % 360 - 180)
            for j in np.argsort(dist + hue_penalty):
                if not np.isfinite(dist[j]) or dist[j] > dist[best] + COLLISION_SLACK:
                    break
                if free(j):
                    choice = int(j)
                    break
        result[i] = int(codes[choice])
        taken.setdefault(int(codes[choice]), centers[i])
    return result


DISTINCT_DE = 25.0       # цвета рисунка дальше этого — разные, им нужен разный пластик
SMALL_FEATURE = 0.05     # доля клеток: только мелкие детали (глаза) уводятся в другой пластик; крупные области берут ближайший
COLLISION_SLACK = 10.0   # насколько дальше ближайшего можно уйти ради различимости (20 уводило крыши в бирюзовый)
HUE_WEIGHT = 0.3         # ΔE за градус оттенка при выборе запасного пластика


MAX_HUE_DIFF = 35  # градусов: цветной пиксель подбирается только среди пластика того же оттенка
MERGE_DE = 7.0     # кластеры ближе этого — один цвет, разбитый освещением (k-means дробит крупные)
GREY_CHROMA = 6.0  # насыщенность в Lab, ниже которой цвет палитры считается серым (Light_Aqua — уже нет)
BASIC_BONUS = 4.0  # ΔE: фора базовым цветам
BASIC_COLORS = {"Black", "White", "Red", "Blue", "Yellow", "Green", "Orange", "Tan", "Dark_Bluish_Gray", "Light_Bluish_Gray",
                "Bright_Pink", "Medium_Azure", "Dark_Blue", "Dark_Red", "Bright_Green", "Lime", "Reddish_Brown", "Dark_Tan",
                "Bright_Light_Orange", "Bright_Light_Blue", "Medium_Blue", "Light_Flesh", "Dark_Pink"}


def _merge_close(centers: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сливает пары ближайших кластеров, пока все центры не разойдутся дальше MERGE_DE."""
    centers, labels = centers.copy(), labels.copy()
    while len(centers) > 1:
        d = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= MERGE_DE:
            break
        ni, nj = (labels == i).sum(), (labels == j).sum()
        centers[i] = (centers[i] * ni + centers[j] * nj) / max(ni + nj, 1)
        labels[labels == j] = i
        labels[labels > j] -= 1
        centers = np.delete(centers, j, axis=0)
    return centers, labels


BLACK = 0
DARK_L, ACHROMATIC = 45.0, 15.0   # DARK_L — для фото: тени на чёрном пластике доходят до L=40
DARK_L_FLAT = 25.0                # у цифрового рисунка серый с L=37 — настоящий цвет (волосы), не тень
DULL_CHROMA = 30.0                # ниже — тусклый цвет, ему разрешён сероватый пластик (Sand_Green, Sand_Blue)
NEAR_BLACK_L = 10.0  # светлота, ниже которой цвет чёрный при любом оттенке: у RGB (15, 1, 42) в Lab
                     # «высокая» насыщенность, но глазом это чёрная обводка, а не Dark_Blue


def _is_black(lab: np.ndarray, dark_l: float | None = None) -> bool:
    dark_l = DARK_L if dark_l is None else dark_l
    return lab[0] < NEAR_BLACK_L or (lab[0] < dark_l and np.hypot(lab[1], lab[2]) < ACHROMATIC)
SAME_HUE_DEG = 20.0  # кластеры одного оттенка (свет и тень одного цвета) сливаются


def _merge_same_hue(lab: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Тень и свет одного цвета — один цвет: сливаем хроматические кластеры с близким оттенком
    в тот, где больше вокселей."""
    centers = np.stack([lab[labels == i].mean(0) if (labels == i).any() else np.zeros(3) for i in range(k)])
    sizes = np.bincount(labels, minlength=k)
    hue = np.degrees(np.arctan2(centers[:, 2], centers[:, 1]))
    chroma = np.hypot(centers[:, 1], centers[:, 2])
    target = np.arange(k)
    for i in np.argsort(sizes):                     # от малых к большим: малый вливается в больший
        if chroma[i] < ACHROMATIC:
            continue
        for j in np.argsort(-sizes):
            if j == i or chroma[j] < ACHROMATIC or sizes[j] <= sizes[i]:
                continue
            if abs((hue[i] - hue[j] + 180) % 360 - 180) < SAME_HUE_DEG:
                target[i] = j
                break
    while not np.array_equal(target[target], target):  # цепочки слияний
        target = target[target]
    return target[labels]
CHROMA_BOOST = 1.4  # фото тусклее пластика: усиливаем насыщенность перед подбором, чтобы выбирались Red/Blue, а не их бледные соседи


def _match_cluster(members: np.ndarray, palette_lab: np.ndarray, codes: np.ndarray) -> int:
    """Тёмное и бесцветное — чёрный (глаза, контуры). Остальное — по светлой половине кластера:
    тени темнее настоящего цвета, блики ближе к нему."""
    center = members.mean(0)
    if _is_black(center):
        return BLACK
    bright = members[members[:, 0] >= np.median(members[:, 0])].mean(0)
    boosted = bright * np.array([1.0, CHROMA_BOOST, CHROMA_BOOST])
    return int(codes[((boosted - palette_lab) ** 2).sum(1).argmin()])


def recolor(codes: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Ручная замена цветов после подбора: {старый код: новый код}."""
    out = codes.copy()
    for old, new in mapping.items():
        out[codes == old] = new
    return out


def code_by_name(name: str) -> int:
    for c in load_palette(common_only=False):
        if c.name.lower() == name.lower() or str(c.code) == name:
            return c.code
    raise ValueError(f"нет такого цвета: {name}")


HSV_HUE_WEIGHT = 0.3   # ΔE за градус расхождения оттенка HSV у насыщенных цветов


def _hsv_hue(rgb: np.ndarray) -> np.ndarray:
    """Оттенок HSV в градусах для массива [n, 3] sRGB 0..255."""
    r, g, b = (np.asarray(rgb, dtype=float) / 255).T
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = np.where(mx - mn == 0, 1, mx - mn)
    hue = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return np.where(mx == mn, 0.0, hue)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (D65) -> sRGB 0..255, с обрезкой в гамут."""
    lab = np.asarray(lab, dtype=float).reshape(-1, 3)
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    def inv(t):
        return np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    xyz = np.c_[inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883]
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = xyz @ m.T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c, 0, 1) * 255


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB, D65."""
    c = rgb.astype(float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)
