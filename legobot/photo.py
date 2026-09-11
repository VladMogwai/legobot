"""Фото -> 3D-меш через публичный Space TRELLIS.2 (Microsoft) на Hugging Face.

Бесплатно, GPU на их стороне. Результат — PLY с цветами вершин, Z вверх,
чтобы дальше по цепочке он ничем не отличался от любого другого меша.
"""
import os
from pathlib import Path

import numpy as np
import trimesh
from gradio_client import Client, handle_file

SPACE = "microsoft/TRELLIS.2"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def hf_token() -> str | None:
    """Токен Hugging Face: из переменной окружения или из .env в корне проекта."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def mesh_from_photo(image_path: str, out_path: str, resolution: int = 1024, seed: int = 0) -> str:
    """Возвращает путь к PLY. resolution — детализация TRELLIS: 512 / 1024 / 1536; seed — вариация."""
    client = Client(SPACE, token=hf_token(), verbose=False)
    client.predict(api_name="/start_session")
    prepared = client.predict(input=handle_file(image_path), api_name="/preprocess_image")
    client.predict(image=handle_file(prepared), seed=seed, resolution=str(resolution), api_name="/image_to_3d")
    glb, _ = client.predict(decimation_target=300_000, texture_size=2048, api_name="/extract_glb")

    mesh = _load_z_up(glb)
    mesh.export(out_path)
    return out_path


def _load_z_up(glb_path: str) -> trimesh.Trimesh:
    """glTF смотрит вверх по Y — поворачиваем в Z-вверх, текстуру переводим в цвета вершин."""
    scene = trimesh.load(glb_path, force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.dump(concatenate=False)]) if hasattr(scene, "dump") else scene
    if hasattr(mesh.visual, "to_color"):
        mesh.visual = mesh.visual.to_color()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh
