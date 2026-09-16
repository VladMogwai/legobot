# legobot

Фото → модель LEGO из настоящих деталей: файл для BrickLink Studio, инструкция PDF, список деталей.

- `python -m legobot photo.jpg --mosaic` — пиксельная фигурка (Pixel Pals и т.п.), стоячая, без 3D.
- `python -m legobot photo.jpg` — объёмная фигурка через фото→3D (HF Space или Kaggle), пластины/кирпичи.
- `python -m legobot model.stl --parts 400` — из 3D-файла.
- `tools/inventory.py model.io` — состав любой модели Studio; `tools/catalog.py` — каталог деталей и цветов.
- `service/` — HTTP-сервис, `web/` — страница с вьюшкой. Подробности и история решений — в `ROADMAP.md`.

Studio нужен только для `--open`; палитра, имена деталей и геометрия — в `catalog/` и `vendor/ldraw/`.
Тесты: `python -m pytest tests`.
