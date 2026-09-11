"""Каталог используемых деталей.

Номера — из библиотеки Studio (/Applications/Studio 2.0/ldraw/parts/).
Размер задан как (по X, по Z) в штырьках в родной ориентации детали.
В LDraw «Brick 2 x 4» лежит длинной стороной вдоль X: 3001 занимает x ±40, z ±20 LDU.
"""
from dataclasses import dataclass

STUD_LDU = 20        # шаг штырьков
BRICK_HEIGHT_LDU = 24
BRICK_ASPECT = BRICK_HEIGHT_LDU / STUD_LDU  # кирпич выше, чем шире


@dataclass(frozen=True)
class Part:
    number: str
    width: int   # по X
    length: int  # по Z

    @property
    def area(self) -> int:
        return self.width * self.length

    @property
    def label(self) -> str:
        return f"{self.width}x{self.length}"


BRICKS = [
    Part("3001", 4, 2),
    Part("3003", 2, 2),
    Part("3010", 4, 1),
    Part("3004", 2, 1),
    Part("3005", 1, 1),
]

PART_BY_SIZE = {(p.width, p.length): p for p in BRICKS}
