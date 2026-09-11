"""Kaggle: фото из датасета legobot-photos -> GLB через TRELLIS.2 на T4.
Запускается пачкой: одна сборка CUDA-расширений (долгая), потом все фото подряд."""
import glob
import os
import subprocess
import sys
import time

t0 = time.time()
os.chdir("/kaggle/working")
run = lambda cmd: subprocess.run(cmd, shell=True, check=True)

# --- 1. TRELLIS.2 и его CUDA-расширения ---
# Единственный подмодуль — Eigen (заголовки) на GitLab, который с Kaggle недоступен, а git к зеркалу
# на GitHub упирается в лимит. Берём архив коммита по HTTPS и кладём на место подмодуля.
EIGEN_SHA = "21e4582d1739107337a03460c81412981130373e"
if not os.path.isdir("TRELLIS.2"):
    run("git clone -q -b main https://github.com/microsoft/TRELLIS.2.git")
os.chdir("TRELLIS.2")
eigen_dir = "o-voxel/third_party/eigen"
if not os.path.exists(os.path.join(eigen_dir, "Eigen")):
    run(f"rm -rf {eigen_dir} && mkdir -p {eigen_dir}")
    run(f"curl -sL https://github.com/eigen-mirror/eigen/archive/{EIGEN_SHA}.tar.gz | tar xz -C {eigen_dir} --strip-components=1")
print("eigen ok:", os.path.exists(os.path.join(eigen_dir, "Eigen", "Dense")), flush=True)
run("bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm > /kaggle/working/setup.log 2>&1 || (tail -50 /kaggle/working/setup.log; exit 1)")
print(f"setup done in {time.time() - t0:.0f}s", flush=True)

# --- 2. T4 не умеет flash-attn: заменяем на встроенный attention ---
patch = "trellis2/modules/attention/full_attn.py"
src = open(patch).read()
if "flash_attn.flash_attn_func(q, k, v)" in src:
    src = src.replace(
        "out = flash_attn.flash_attn_func(q, k, v)",
        "import torch.nn.functional as F; out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)",
    )
    open(patch, "w").write(src)
os.environ["ATTN_BACKEND"] = "math"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/kaggle/working/TRELLIS.2")

# --- 3. Пайплайн ---
import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
# 4B параметров в fp32 не влезают в 16 ГБ T4 — переводим в fp16
for name, model in getattr(pipeline, "models", {}).items():
    try:
        model.half()
    except Exception as e:  # noqa: BLE001
        print("fp16 skip", name, e)
print(f"pipeline ready at {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

# --- 4. Параметры прогона: config.json в датасете (seeds, resolution) ---
import json
config = {"seeds": [0], "resolution": "1024"}
if os.path.exists("/kaggle/input/legobot-photos/config.json"):
    config.update(json.load(open("/kaggle/input/legobot-photos/config.json")))
RESOLUTION = str(config["resolution"])
PIPELINE_TYPE = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[RESOLUTION]
os.makedirs("/kaggle/working/output", exist_ok=True)

photos = sorted(glob.glob("/kaggle/input/legobot-photos/*.*"))
photos = [p for p in photos if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
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
