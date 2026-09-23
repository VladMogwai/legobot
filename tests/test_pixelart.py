"""Эталоны: фото пиксельных фигурок. Любое изменение сетки или подбора цвета должно пройти
через них: если хоть одна фигурка «поплыла», изменение не принимается."""
from pathlib import Path

import numpy as np
import pytest

from legobot.colors import _rgb_to_lab, studio_palette
from legobot.mosaic import standing_bricks
from legobot.pixelart import mosaic_from_photo

# файл, сетка (ш×в), клеток (±5 %), цвета, которые обязаны быть, средняя ΔE цветных клеток не выше
REFERENCE = [
    ("photos/mario.jpeg", (20, 30), 423, {"Black", "White", "Red", "Blue"}, 16),
    ("examples/vaultboy.webp", (24, 23), 321, {"Black", "Blue", "Bright_Light_Yellow"}, 16),
    ("examples/mario8.png", (18, 26), 322, {"Black", "White", "Red"}, 20),
    ("examples/megaman.webp", (23, 24), 360, {"Black", "White", "Medium_Azure"}, 23),   # тени на контуре — чёрные; неоновый голубой дальше Medium_Azure нет
    ("examples/cow.png", (54, 54), 542, {"Black", "White"}, 18),   # розовое вымя: Pink на Pick a Brick нет   # рисунок: пиксель 17.6 px, спрайт 55 клеток
    ("examples/purple-phantom.png", (46, 54), 1377, {"Black", "Dark_Purple"}, 18),   # обводка тёмно-лиловая (15, 1, 42) — чёрная; сетка со «швом»
    ("examples/dragon.webp", (42, 35), 888, {"Black", "Green", "Yellow"}, 14),   # фото на бежевом: rembg; жёлтый глаз — дыра в маске
    # стоковые картинки (examples/stock не в git — без файла тест пропускается)
    ("examples/stock/girl.jpg", (50, 53), 343, {"Black", "Orange", "Medium_Azure"}, 17),        # мелкий спрайт на большом белом поле
    ("examples/stock/purpleshirt.jpg", (31, 31), 289, {"Medium_Lavender", "Dark_Bluish_Gray"}, 14),   # без контура; серые волосы — не чёрные
    ("examples/stock/redcap.jpg", (23, 22), 118, {"Red", "Dark_Blue", "Medium_Dark_Flesh"}, 11),   # без контура; симметричная
    ("examples/stock/blueskin.jpg", (85, 76), 2516, {"Black", "Light_Aqua", "Red"}, 16),          # «векторный» пиксель-арт с неровными блоками — выравнивается по симметрии
    ("examples/stock/dog1.jpg", (61, 61), 1129, {"White", "Black", "Light_Flesh"}, 17),          # белое тело на белом фоне внутри контура — полая фигура заливается
    ("examples/stock/robot2.jpg", (87, 88), 1954, {"Black", "White"}, 13),                       # то же; крупная (87 клеток)
    ("examples/stock/fox.webp", (45, 48), 1067, {"Orange", "Medium_Blue"}, 18),                  # шахматный фон «прозрачности»
    ("examples/stock/eevee.png", (39, 43), 810, {"Flesh", "Black", "Reddish_Brown"}, 14),           # PNG с альфой на шахматке
    ("examples/stock/car1.jpg", (41, 29), 475, {"Dark_Azure", "Black"}, 14),                    # техника в профиль
    ("examples/stock/mew.png", (46, 44), 750, {"Black", "Bright_Pink", "Dark_Pink"}, 20),   # два розовых остаются разными
    ("examples/stock/sonic1.png", (32, 45), 569, {"Blue", "Black"}, 20),                          # Genesis, PNG с прозрачным фоном: маска — альфа
    ("examples/stock/samus1.png", (25, 49), 834, {"Red", "Black", "Orange"}, 16),          # SNES
    ("examples/stock/samus3.png", (16, 32), 307, {"Red", "Orange", "Green"}, 20),                 # NES: мелкая (16 клеток)
    ("examples/stock/pixilart.png", (64, 64), 426, {"Black", "Sand_Green", "Flesh"}, 13),        # цветной фон; тусклый зелёный → Sand_Green
]


@pytest.fixture(scope="module", params=REFERENCE, ids=[r[0].split("/")[-1] for r in REFERENCE])
def case(request):
    path, size, cells, colors, max_de = request.param
    if not Path(path).exists():
        pytest.skip(f"{path} нет (стоковые картинки не в git)")
    return mosaic_from_photo(path), size, cells, colors, max_de


def test_grid(case):
    result, size, cells, _, _ = case
    m = result.mosaic
    assert abs(m.width - size[0]) <= 1 and abs(m.height - size[1]) <= 1, (m.width, m.height)   # ±1 ряд по краю — фаза
    assert abs(int((m.codes >= 0).sum()) - cells) <= cells * 0.05


def test_colors(case):
    result, _, _, expected, max_de = case
    m, photo = result.mosaic, result.cell_colors
    names = {c.code: c.name for c in studio_palette(common_only=False)}
    used = {names[int(c)] for c in np.unique(m.codes[m.codes >= 0])}
    assert expected <= used, used
    lego = {c.code: c.rgb for c in studio_palette(common_only=False)}
    present = m.codes >= 0
    chosen = np.array([lego[int(c)] for c in m.codes[present]], dtype=np.uint8)
    lab = _rgb_to_lab(photo[present])
    de = np.sqrt(((lab - _rgb_to_lab(chosen)) ** 2).sum(1))
    colored = np.hypot(*lab[:, 1:].T) > 15
    assert de[colored].mean() < max_de, de[colored].mean()   # контур намеренно чёрный, его не считаем


def test_standing_is_one_piece(case):
    result, _, cells, _, _ = case
    bricks = standing_bricks(result.mosaic)
    assert cells * 0.4 <= len(bricks) <= cells * 1.2   # большие однотонные поля сливаются в 2×8
    grid = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.width):
            for z in range(b.z, b.z + b.length):
                grid[(x, z, b.layer)] = i
    parent = list(range(len(bricks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for (x, z, k), i in grid.items():
        j = grid.get((x, z, k + 1))
        if j is not None:
            parent[find(i)] = find(j)
    assert len({find(i) for i in range(len(bricks))}) == 1


# Панно — фотография выложенной картины: свой подбор цвета (panel_codes), своя проверка.
# файл, сетка, цветов не меньше, средняя ΔE не выше, чёрных клеток и одиночных — не больше доли
PANELS = [
    ("examples/stock/starry2.jpg", (130, 168), 25, 14.5, 0.20, 0.07),   # фото мозаики 128×166: мелкие места набраны 1×1, сетка дробится
    ("examples/purple-phantom.png", (46, 54), 4, 10.0, 0.60, 0.05),     # рисунок панно: контур намеренно чёрный
]


@pytest.fixture(scope="module", params=PANELS, ids=[p[0].split("/")[-1] for p in PANELS])
def panel(request):
    path, *rest = request.param
    if not Path(path).exists():
        pytest.skip(f"{path} нет (стоковые картинки не в git)")
    return (mosaic_from_photo(path, max_colors=32, keep_background=True), *rest)


def test_panel(panel):
    result, size, min_colors, max_de, max_black, _ = panel
    m, photo = result.mosaic, result.cell_colors
    assert (m.width, m.height) == size, (m.width, m.height)
    present = m.codes >= 0
    used = m.codes[present]
    assert len(np.unique(used)) >= min_colors, len(np.unique(used))
    lego = {c.code: c.rgb for c in studio_palette(common_only=False)}
    black = next(c.code for c in studio_palette(common_only=False) if c.name == "Black")
    # тёмное у картины — не контур: если «всё тёмное в чёрный», ночное небо и крыши пропадают
    assert (used == black).mean() <= max_black, (used == black).mean()
    chosen = np.array([lego[int(c)] for c in used], dtype=np.uint8)
    de = np.sqrt(((_rgb_to_lab(photo[present]) - _rgb_to_lab(chosen)) ** 2).sum(1))
    assert de.mean() < max_de, de.mean()


def test_panel_has_no_speckle(panel):
    """Клетка, не совпавшая ни с одним из четырёх соседей, — крапина: лишняя деталь и шум.
    Совсем до нуля не свести: у «Звёздной ночи» около 5 % — это мазки самой картины (звёзды,
    окна в один пиксель). Без оглядки на соседей при подборе цвета было 13 %."""
    m = panel[0].mosaic
    codes = m.codes
    lone = np.zeros(codes.shape, bool)
    lone[1:-1, 1:-1] = ((codes[1:-1, 1:-1] != codes[:-2, 1:-1]) & (codes[1:-1, 1:-1] != codes[2:, 1:-1])
                        & (codes[1:-1, 1:-1] != codes[1:-1, :-2]) & (codes[1:-1, 1:-1] != codes[1:-1, 2:]))
    assert lone.mean() < panel[5], lone.mean()
