"""python tools/inventory.py <model.io|.ldr|.mpd> — состав модели: детали, цвета, категории."""
import sys

from legobot.inventory import inventory, read_model, report

print(report(inventory(read_model(sys.argv[1]))))
