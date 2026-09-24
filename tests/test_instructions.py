"""Инструкция должна рисоваться по любой модели, в том числе по чужой из Studio."""
from dataclasses import replace

import numpy as np

from legobot.instructions import bill_of_materials, split_steps, step_size, write_pdf
from legobot.mosaic import Mosaic, standing_bricks


def _bricks():
    codes = np.full((6, 6), 0)
    codes[2:4, 2:4] = 4
    return standing_bricks(Mosaic(codes, 1.0, 6, 6))


def test_guide_draws_colours_outside_the_palette(tmp_path):
    """Прозрачные и металлики бот сам не кладёт, но из чужого .io они приходят — и деталь
    неизвестного цвета не должна ронять рендер."""
    bricks = _bricks()
    bricks[0] = replace(bricks[0], color=47)      # Trans_Clear
    bricks[1] = replace(bricks[1], color=9999)    # такого цвета нет в LDraw вообще
    pages = write_pdf(bricks, split_steps(bricks), str(tmp_path / "guide.pdf"), "проверка")
    assert pages > 1 and (tmp_path / "guide.pdf").stat().st_size > 10_000
    names = {l.color_name for l in bill_of_materials(bricks)}
    assert "Trans_Clear" in names and "9999" in names   # неизвестный остаётся кодом, но не падает


def test_step_grows_with_the_model():
    """Шаг растёт вместе с моделью: иначе страницы перерисовывают её сотни раз и PDF не открыть."""
    small = _bricks()
    big = small * 40
    assert step_size(small) < step_size(big)
