"""Модель должна держаться одним куском и стоять — это проверяется на цифрах, а не при сборке."""
import numpy as np
import pytest

from legobot.inflate import volume_bricks
from legobot.mosaic import Mosaic, mosaic_bricks, standing_bricks
from legobot.pixelart import mosaic_from_image, mosaic_from_photo
from legobot.stability import check


@pytest.fixture(scope="module")
def mosaic():
    return mosaic_from_photo("photos/mario.jpeg").mosaic


@pytest.fixture(scope="module")
def panel():
    return mosaic_from_image("photos/mario.jpeg", 32, keep_background=True).mosaic


def test_figure_holds_together(mosaic):
    for bricks in (standing_bricks(mosaic), volume_bricks(mosaic, 4)):
        report = check(bricks)
        assert report.pieces == 1 and report.loose == 0, report.problems


def test_panel_with_base_holds_together(panel):
    """Подложка и плитки кладутся одним проходом, и плитки садятся так, чтобы сшивать пластины:
    панно выходит одним куском. Двумя проходами и подложкой из пластин 6×6 оно разваливалось
    на тринадцать частей."""
    report = check(mosaic_bricks(panel, base=True))
    assert report.pieces == 1 and report.loose == 0, report.problems


def test_panel_of_lone_cells_is_reported(panel):
    """Если каждая клетка своего цвета, стык пластин перекрыть нечем — ни одна плитка не ложится
    на две сразу. Собрать такое панно единым куском нельзя, и проверка обязана об этом сказать."""
    speckled = Mosaic(np.arange(32 * 32).reshape(32, 32) % 5, 1.0, 32, 32)
    report = check(mosaic_bricks(speckled, base=True))
    assert report.pieces > 1 and any("распадается" in p for p in report.problems)


def test_panel_without_base_is_laid_on_a_baseplate(panel):
    """Без своей подложки плитки и не должны быть связаны: их держит готовая пластина."""
    report = check(mosaic_bricks(panel, base=False))
    assert report.on_baseplate and report.ok


def test_thin_figure_is_reported_as_tippy(mosaic):
    assert check(standing_bricks(mosaic)).ok
    assert not check(standing_bricks(mosaic, depth=2)).ok    # вдвое тоньше — уже валится
