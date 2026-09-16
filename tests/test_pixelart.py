"""Эталон: Марио с фото photos/mario.jpeg. Любое изменение сетки или подбора цвета
должно пройти через него: если Марио «поплыл», изменение не принимается."""
import numpy as np
import pytest

from legobot.colors import _rgb_to_lab, studio_palette
from legobot.mosaic import standing_bricks
from legobot.pixelart import mosaic_from_photo

EXPECTED_COLORS = {"Black", "White", "Red", "Blue", "Bright_Light_Blue", "Medium_Orange", "Bright_Pink", "Dark_Pink", "Light_Aqua", "Yellow"}


@pytest.fixture(scope="module")
def mario():
    return mosaic_from_photo("photos/mario.jpeg")


def test_grid(mario):
    m = mario.mosaic
    assert (m.width, m.height) == (20, 30)
    assert 355 <= int((m.codes >= 0).sum()) <= 370


def test_colors(mario):
    names = {c.code: c.name for c in studio_palette(common_only=False)}
    used = {names[int(c)] for c in np.unique(mario.mosaic.codes[mario.mosaic.codes >= 0])}
    assert used == EXPECTED_COLORS, used


def test_color_accuracy(mario):
    m, cells = mario.mosaic, mario.cell_colors
    lego = {c.code: c.rgb for c in studio_palette(common_only=False)}
    present = m.codes >= 0
    chosen = np.array([lego[int(c)] for c in m.codes[present]], dtype=np.uint8)
    de = np.sqrt(((_rgb_to_lab(cells[present]) - _rgb_to_lab(chosen)) ** 2).sum(1))
    colored = np.hypot(*_rgb_to_lab(cells[present])[:, 1:].T) > 15
    assert de[colored].mean() < 16, de[colored].mean()   # контур намеренно чёрный, его не считаем


def test_standing_is_one_piece(mario):
    bricks = standing_bricks(mario.mosaic)
    assert 250 <= len(bricks) <= 320
    cells = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.width):
            for z in range(b.z, b.z + b.length):
                cells[(x, z, b.layer)] = i
    parent = list(range(len(bricks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for (x, z, k), i in cells.items():
        j = cells.get((x, z, k + 1))
        if j is not None:
            parent[find(i)] = find(j)
    assert len({find(i) for i in range(len(bricks))}) == 1
