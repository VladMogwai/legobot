"""Выложить сайт: python tools/deploy_web.py

Собирает движок (tools/web_engine.py) и загружает web/public целиком на статический Space
Hugging Face — бесплатный хостинг статики; расчёт идёт в браузере, сервер не нужен.
Токен — HF_TOKEN_WRITE в .env (в чат и в git не попадает).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = "vmogwai/legobot"
PUBLIC = ROOT / "web" / "public"


def token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN_WRITE="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("нет HF_TOKEN_WRITE в .env")


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "tools" / "web_engine.py")], check=True, cwd=ROOT)
    from huggingface_hub import HfApi
    HfApi(token=token()).upload_folder(folder_path=str(PUBLIC), repo_id=SPACE, repo_type="space",
                                       commit_message="site: in-browser engine", delete_patterns=["engine/*", "demo/*"])
    print(f"-> https://{SPACE.replace('/', '-')}.static.hf.space/")


if __name__ == "__main__":
    main()
