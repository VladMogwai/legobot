"""Палитра цветов LDraw из библиотеки Studio и подбор ближайшего цвета."""
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

LDCONFIG_PATH = "/Applications/Studio 2.0/ldraw/LDConfig.ldr"

# Материалы, которые не годятся для обычных кирпичей: прозрачные, металлики, резина и т.п.
_SPECIAL = ("ALPHA", "CHROME", "PEARLESCENT", "RUBBER", "MATTE_METALLIC", "METAL", "MATERIAL", "LUMINANCE")

_LINE = re.compile(r"^0 !COLOUR (\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})")


@dataclass(frozen=True)
class LdrawColor:
    code: int
    name: str
    rgb: tuple[int, int, int]


@lru_cache
def load_palette() -> tuple[LdrawColor, ...]:
    """Сплошные цвета из LDConfig.ldr, в порядке файла."""
    palette = []
    with open(LDCONFIG_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _LINE.match(line)
            if not m or any(tag in line for tag in _SPECIAL):
                continue
            name, code, hexrgb = m.groups()
            rgb = tuple(int(hexrgb[i:i + 2], 16) for i in (0, 2, 4))
            palette.append(LdrawColor(int(code), name, rgb))
    return tuple(palette)


def nearest_codes(rgb: np.ndarray) -> np.ndarray:
    """rgb: uint8 [..., 3] -> коды LDraw той же формы (ближайший цвет по RGB)."""
    palette = load_palette()
    table = np.array([c.rgb for c in palette], dtype=float)
    codes = np.array([c.code for c in palette])
    flat = rgb.reshape(-1, 3).astype(float)
    dist = ((flat[:, None, :] - table[None, :, :]) ** 2).sum(-1)
    return codes[dist.argmin(1)].reshape(rgb.shape[:-1])
