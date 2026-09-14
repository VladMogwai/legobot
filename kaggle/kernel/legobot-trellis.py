"""Kaggle: фото из датасета legobot-photos -> GLB через TRELLIS.2 на T4.
Запускается пачкой: одна сборка CUDA-расширений (долгая), потом все фото подряд."""
import glob
import os
import subprocess
import sys
import time

t0 = time.time()
# Kaggle монтирует датасеты в /kaggle/input/datasets/<владелец>/<слаг>/ (раньше — /kaggle/input/<слаг>/).
INPUT = next((p for p in ("/kaggle/input/datasets/vladmogwai/legobot-photos", "/kaggle/input/legobot-photos") if os.path.isdir(p)), None)
if INPUT is None:
    raise SystemExit("датасет legobot-photos не смонтирован")
print("input:", INPUT, sorted(os.listdir(INPUT)), flush=True)
# Всё тяжёлое — во временную папку: Kaggle сохраняет /kaggle/working как результат целиком.
WORK = "/kaggle/tmp"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
run = lambda cmd: subprocess.run(cmd, shell=True, check=True)

# --- 1. TRELLIS.2 и его CUDA-расширения ---
# Единственный подмодуль — Eigen (заголовки) на GitLab, который с Kaggle недоступен, а git к зеркалу
# на GitHub упирается в лимит. Берём архив коммита по HTTPS и кладём на место подмодуля.
EIGEN_SHA = "21e4582d1739107337a03460c81412981130373e"
if not os.path.isdir("TRELLIS.2"):
    run("git clone -q -b main https://github.com/microsoft/TRELLIS.2.git")
os.chdir(f"{WORK}/TRELLIS.2")
eigen_dir = "o-voxel/third_party/eigen"
if not os.path.exists(os.path.join(eigen_dir, "Eigen")):
    run(f"rm -rf {eigen_dir} && mkdir -p {eigen_dir}")
    run(f"curl -sL https://github.com/eigen-mirror/eigen/archive/{EIGEN_SHA}.tar.gz | tar xz -C {eigen_dir} --strip-components=1")
print("eigen ok:", os.path.exists(os.path.join(eigen_dir, "Eigen", "Dense")), flush=True)
# flash-attn на Kaggle собирается из исходников ~9 часов и на T4 всё равно не работает.
# Вместо него xformers: готовое колесо, и его понимают обе части TRELLIS — плотная и разреженная.
# В образе Kaggle torch 2.10 (cu128); xformers берём с PyPI под него, torch не трогаем —
# иначе setup.sh вернёт torch обратно, а ядра xformers останутся под старым.
run("pip install -q 'xformers>=0.0.35' > /kaggle/working/setup.log 2>&1")
run("""python - <<'PY'
import torch, xformers, xformers.ops as xops
print("torch", torch.__version__, "xformers", xformers.__version__, "GPU", torch.cuda.get_device_name(0))
q = torch.randn(1, 4096, 12, 128, device="cuda", dtype=torch.float16)
out = xops.memory_efficient_attention(q, q, q)   # то, что зовёт TRELLIS; на T4 должен сработать cutlass
print("attention ok", tuple(out.shape))
PY""")
run("bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm >> /kaggle/working/setup.log 2>&1 || (tail -80 /kaggle/working/setup.log; exit 1)")
run("python -c 'import o_voxel, nvdiffrast, cumesh; print(\"extensions ok\")'")
print(f"setup done in {time.time() - t0:.0f}s", flush=True)

# --- 2. Бэкенд внимания ---
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, f"{WORK}/TRELLIS.2")

# --- 3. Логин в Hugging Face: энкодер DINOv3 у TRELLIS.2 закрытый, нужен токен с принятыми условиями ---
# Kaggle Secrets из запусков через API недоступны, поэтому токен приходит файлом в приватном датасете.
from huggingface_hub import login
token = None
token_file = f"{INPUT}/hf_token.txt"
if os.path.exists(token_file):
    token = open(token_file).read().strip()
else:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF")
    except Exception as e:  # noqa: BLE001
        print("HF secret unavailable:", e, flush=True)
if token:
    login(token=token)
    print("HF login ok", flush=True)
else:
    print("HF login skipped: no token", flush=True)

# --- 4. Пайплайн ---
import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Модель вырезания фона (briaai/RMBG-2.0) закрытая и с ручным одобрением. Фон вырезан заранее,
# фото приходят с альфа-каналом — при нём пайплайн свою модель не вызывает; подменяем её заглушкой.
import trellis2.pipelines.rembg as trellis_rembg


class _NoRembg:
    def __init__(self, *a, **k): pass
    def to(self, *a, **k): return self
    def cpu(self): return self
    def __call__(self, image):
        raise RuntimeError("фото без альфа-канала: фон должен быть вырезан до отправки")


trellis_rembg.BiRefNet = _NoRembg

# Модель переведена в fp16, а часть активаций приходит в fp32; Triton-ядро разреженной свёртки
# (flex_gemm) требует одинаковые типы — приводим вход к типу весов перед вызовом.
import trellis2.modules.sparse.conv.conv_flex_gemm as conv_flex_gemm
_orig_conv_forward = conv_flex_gemm.sparse_conv3d_forward


def _conv_forward_same_dtype(self, x):
    if x.feats.dtype != self.weight.dtype:
        x = x.replace(x.feats.to(self.weight.dtype))
    return _orig_conv_forward(self, x)


conv_flex_gemm.sparse_conv3d_forward = _conv_forward_same_dtype
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
# 4B параметров в fp32 не влезают в 16 ГБ T4 — переводим в fp16
for name, model in getattr(pipeline, "models", {}).items():
    try:
        model.half()
    except Exception as e:  # noqa: BLE001
        print("fp16 skip", name, e)
print(f"pipeline ready at {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

# --- 5. Параметры прогона: config.json в датасете (seeds, resolution) ---
import json
config = {"seeds": [0], "resolution": "1024"}
if os.path.exists(f"{INPUT}/config.json"):
    config.update(json.load(open(f"{INPUT}/config.json")))
RESOLUTION = str(config["resolution"])
PIPELINE_TYPE = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[RESOLUTION]
os.makedirs("/kaggle/working/output", exist_ok=True)

photos = sorted(glob.glob(f"{INPUT}/*.*"))
photos = [p for p in photos if p.lower().endswith(".png")]  # только PNG с вырезанным фоном
print("photos:", photos, "seeds:", config["seeds"], "resolution:", RESOLUTION, flush=True)

for path in photos:
    name = os.path.splitext(os.path.basename(path))[0]
    image = pipeline.preprocess_image(Image.open(path))
    for seed in config["seeds"]:
        t1 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = pipeline.run(image, seed=int(seed), preprocess_image=False, pipeline_type=PIPELINE_TYPE)
            mesh = outputs[0]
            # экспорт ждёт fp32, а после перевода модели в fp16 атрибуты приходят в half
            for attr in ("vertices", "faces", "attrs", "coords"):
                t = getattr(mesh, attr, None)
                if t is not None and t.dtype == torch.float16:
                    setattr(mesh, attr, t.float())
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout, grid_size=int(RESOLUTION),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=300_000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
            )
            glb.export(f"/kaggle/working/output/{name}_s{seed}.glb", extension_webp=True)
            print(f"OK {name} seed {seed} in {time.time() - t1:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"FAIL {name} seed {seed}: {e}", flush=True)
        torch.cuda.empty_cache()

print(f"all done in {time.time() - t0:.0f}s", flush=True)
