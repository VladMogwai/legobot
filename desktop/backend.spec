# PyInstaller: бэкенд legobot одной папкой (dist/legobot-server). Запуск: pyinstaller desktop/backend.spec
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]
for pkg in ("rembg", "onnxruntime", "skimage", "scipy", "matplotlib", "trimesh", "pooch", "PIL", "numpy", "pymatting"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
for pkg in ("pymatting", "rembg", "onnxruntime", "pooch", "numpy", "scipy", "scikit-image", "pillow", "trimesh", "matplotlib", "numba", "llvmlite"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
datas += [
    (str(ROOT / "catalog"), "catalog"),
    (str(ROOT / "vendor" / "ldraw"), "vendor/ldraw"),
    (str(ROOT / "web" / "public"), "web/public"),
    (str(ROOT / "legobot.toml"), "."),
    (str(ROOT / "desktop" / "rembg_home"), "rembg_home"),
]

a = Analysis([str(ROOT / "service" / "run_server.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, excludes=["tkinter", "pytest", "IPython"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="legobot-server", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="legobot-server")
