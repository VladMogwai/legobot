"""Каталог используемых деталей.

Номера — из библиотеки Studio (/Applications/Studio 2.0/ldraw/parts/).
Размер задан как (по X, по Z) в штырьках в родной ориентации детали.
В LDraw «2 x 4» лежит длинной стороной вдоль X: 3001 занимает x ±40, z ±20 LDU.
"""
from dataclasses import dataclass

STUD_LDU = 20         # шаг штырьков
BRICK_HEIGHT_LDU = 24
PLATE_HEIGHT_LDU = 8  # пластина — треть кирпича


@dataclass(frozen=True)
class Part:
    number: str
    width: int   # по X
    length: int  # по Z
    height: int  # LDU

    @property
    def area(self) -> int:
        return self.width * self.length

    @property
    def label(self) -> str:
        return f"{self.length}x{self.width}"  # как называет LEGO: 2x4, а не 4x2


@dataclass(frozen=True)
class Vocabulary:
    """Набор деталей одной высоты, из которых ведётся кладка."""
    name: str
    parts: tuple[Part, ...]

    @property
    def height(self) -> int:
        return self.parts[0].height

    @property
    def aspect(self) -> float:
        """Во сколько раз деталь выше шага штырьков: 1.2 у кирпича, 0.4 у пластины."""
        return self.height / STUD_LDU

    def by_size(self, width: int, length: int) -> Part | None:
        return next((p for p in self.parts if (p.width, p.length) == (width, length)), None)


def _bricks(*spec):
    return tuple(Part(n, w, l, BRICK_HEIGHT_LDU) for n, w, l in spec)


def _plates(*spec):
    return tuple(Part(n, w, l, PLATE_HEIGHT_LDU) for n, w, l in spec)


BRICKS = Vocabulary("bricks", _bricks(
    ("3007", 8, 2), ("2456", 6, 2), ("3001", 4, 2), ("3003", 2, 2),
    ("3008", 8, 1), ("3009", 6, 1), ("3010", 4, 1), ("3004", 2, 1), ("3005", 1, 1),
))

PLATES = Vocabulary("plates", _plates(
    ("3034", 8, 2), ("3795", 6, 2), ("3020", 4, 2), ("3022", 2, 2),
    ("3460", 8, 1), ("3666", 6, 1), ("3710", 4, 1), ("3023", 2, 1), ("3024", 1, 1),
))

VOCABULARIES = {v.name: v for v in (BRICKS, PLATES)}
