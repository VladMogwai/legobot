"""Проверка модели на прочность: держится ли она одним куском и устоит ли на столе.

Собирают руками, и узнать, что фигурка рассыпается или падает, обычно удаётся только с деталями
в руках. Здесь то же самое считается по модели — до того, как напечатана инструкция и куплены
детали. Связь между деталями одна: штырьки, то есть перекрытие клеток соседних слоёв.
"""
from collections import defaultdict
from dataclasses import dataclass

STUDS_TO_HOLD = 2     # деталь на одном штырьке крутится и отваливается; деталь 1×1 — исключение,
                      # больше одного штырька ей взять неоткуда
TIPPY = 12.0          # во сколько раз высота может превышать глубину опоры: при толщине по
                      # умолчанию (4 штырька) фигурка стоит, вдвое тоньше — уже валится


@dataclass(frozen=True)
class Report:
    pieces: int              # на сколько не связанных между собой кусков распадается модель
    loose: int               # деталей, которым не хватает штырьков, чтобы держаться
    slenderness: float       # высота / глубина опоры: чем больше, тем легче уронить
    on_baseplate: bool       # модель в один слой: её кладут на готовую строительную пластину
    problems: list[str]      # то же словами; пусто — претензий нет

    @property
    def ok(self) -> bool:
        return not self.problems


def check(bricks: list) -> Report:
    """Отчёт о прочности. Панно без своей подложки — один слой: его держит чужая пластина,
    поэтому ни связность, ни штырьки с него не спрашиваем."""
    on_baseplate = len({b.layer for b in bricks}) == 1
    holds = _holds(bricks)
    pieces = _pieces(bricks, holds)
    loose = sum(1 for i, b in enumerate(bricks) if holds[i] < min(b.width * b.length, STUDS_TO_HOLD))
    slenderness = _slenderness(bricks)
    problems = []
    if not on_baseplate:
        if pieces > 1:
            problems.append(f"модель распадается на {pieces} не связанных между собой куска — их нечем скрепить")
        if loose:
            problems.append(f"деталей на одном штырьке или вовсе без опоры: {loose}")
    if slenderness > TIPPY:
        problems.append(f"фигурка узкая: высота больше глубины в {slenderness:.0f} раз — поставь её к опоре "
                        f"или собери толще")
    return Report(pieces, loose, slenderness, on_baseplate, problems)


def _holds(bricks: list) -> dict[int, int]:
    """Сколько штырьков держит каждую деталь: клетки, которыми она перекрывается с соседними слоями."""
    occupied = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.width):
            for z in range(b.z, b.z + b.length):
                occupied[(x, z, b.layer)] = i
    holds: dict[int, int] = defaultdict(int)
    for (x, z, k), i in occupied.items():
        j = occupied.get((x, z, k + 1))
        if j is not None:
            holds[i] += 1
            holds[j] += 1
    return holds


def _pieces(bricks: list, holds: dict[int, int]) -> int:
    occupied = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.width):
            for z in range(b.z, b.z + b.length):
                occupied[(x, z, b.layer)] = i
    parent = list(range(len(bricks)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (x, z, k), i in occupied.items():
        j = occupied.get((x, z, k + 1))
        if j is not None:
            parent[find(i)] = find(j)
    return len({find(i) for i in range(len(bricks))})


def _slenderness(bricks: list) -> float:
    """Высота к глубине опоры в одних единицах: у лежащего панно опора больше высоты, у стоячей
    фигурки высота в разы больше — такую роняет сквозняк."""
    studs = 20.0     # LDU в штырьке по горизонтали
    depth = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
    height = (max(b.layer for b in bricks) + 1) * bricks[0].part.height / studs
    return height / max(depth, 1)
