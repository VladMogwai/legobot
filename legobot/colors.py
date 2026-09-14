"""Палитра цветов LDraw из библиотеки Studio и подбор цветов модели.

Цвета сравниваются в CIELAB — расстояние там соответствует тому, как видит глаз.
Цвета модели сначала сводятся к нескольким доминирующим (k-means): у модели
из деталей цветов мало, а тени и блики с фото — не цвета.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.cluster.vq import kmeans2

LDCONFIG_PATH = "/Applications/Studio 2.0/ldraw/LDConfig.ldr"

# Материалы, которые не годятся для обычных деталей: прозрачные, металлики, резина и т.п.
_SPECIAL = ("ALPHA", "CHROME", "PEARLESCENT", "RUBBER", "MATTE_METALLIC", "METAL", "MATERIAL", "LUMINANCE")

# Ходовые сплошные цвета: то, что реально есть в продаже в любых деталях.
COMMON_COLORS = {
    0, 1, 2, 4, 14, 15, 19, 25, 27, 28, 70, 71, 72, 73, 74, 84, 85, 191, 212, 226, 272, 288, 308, 320, 321, 322, 323, 326, 378, 379, 462, 484,
    5, 13, 22, 26, 29, 30, 31, 69, 112,   # розовые, пурпурные, лиловые
    78, 86, 92,                           # телесные: Light_Flesh, Dark_Flesh, Flesh
}

_LINE = re.compile(r"^0 !COLOUR (\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})")


@dataclass(frozen=True)
class LdrawColor:
    code: int
    name: str
    rgb: tuple[int, int, int]


@lru_cache
def load_palette(common_only: bool = True) -> tuple[LdrawColor, ...]:
    """Сплошные цвета из LDConfig.ldr, в порядке файла."""
    palette = []
    with open(LDCONFIG_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _LINE.match(line)
            if not m or any(tag in line for tag in _SPECIAL):
                continue
            name, code, hexrgb = m.groups()
            if common_only and int(code) not in COMMON_COLORS:
                continue
            rgb = tuple(int(hexrgb[i:i + 2], 16) for i in (0, 2, 4))
            palette.append(LdrawColor(int(code), name, rgb))
    return tuple(palette)


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
    palette = load_palette()
    palette_lab = _rgb_to_lab(np.array([c.rgb for c in palette]))
    codes = np.array([c.code for c in palette])
    center_codes = np.zeros(k, dtype=int)
    for i in range(k):
        members = lab[labels == i] if (labels == i).any() else lab
        center_codes[i] = _match_cluster(members, palette_lab, codes)
    return center_codes[labels]


BLACK = 0
DARK_L, ACHROMATIC = 45.0, 15.0


def _match_cluster(members: np.ndarray, palette_lab: np.ndarray, codes: np.ndarray) -> int:
    """Тёмное и бесцветное — чёрный (глаза, контуры). Остальное — по светлой половине кластера:
    тени темнее настоящего цвета, блики ближе к нему."""
    center = members.mean(0)
    if center[0] < DARK_L and np.hypot(center[1], center[2]) < ACHROMATIC:
        return BLACK
    bright = members[members[:, 0] >= np.median(members[:, 0])].mean(0)
    return int(codes[((bright - palette_lab) ** 2).sum(1).argmin()])


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


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB, D65."""
    c = rgb.astype(float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)
