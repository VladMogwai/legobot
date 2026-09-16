"""Настройки вкуса из legobot.toml в корне проекта: то, что нельзя вычислить из входных данных."""
import tomllib
from pathlib import Path

DEFAULTS = {"mosaic": {"outline_black": True, "back_color": "Dark_Bluish_Gray", "palette": "common"}}
PATH = Path(__file__).resolve().parent.parent / "legobot.toml"


def preferences() -> dict:
    prefs = {section: dict(values) for section, values in DEFAULTS.items()}
    if PATH.exists():
        for section, values in tomllib.loads(PATH.read_text()).items():
            prefs.setdefault(section, {}).update(values)
    return prefs
