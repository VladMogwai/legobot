"""Запуск сервиса на этом Mac и вывод его наружу: python tools/serve.py

1. uvicorn с service.app на порту 7860;
2. cloudflared quick tunnel (без регистрации; адрес *.trycloudflare.com, новый при каждом запуске);
3. адрес записывается в web/public/config.js и заливается на статический Space,
   чтобы страница сразу смотрела на живой бэкенд.
Ctrl+C останавливает всё. Пока ноут выключен или скрипт не запущен — страница покажет «сервис офлайн».
"""
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 7860
SPACE = "vmogwai/legobot"
CONFIG = ROOT / "web" / "public" / "config.js"
CLOUDFLARED = next((p for p in ("/opt/homebrew/bin/cloudflared", "/opt/homebrew/opt/cloudflared/bin/cloudflared", "/usr/local/bin/cloudflared") if os.path.exists(p)), "cloudflared")


def main() -> None:
    if _port_busy():
        print(f"порт {PORT} занят — сервис уже запущен в другом окне. Останови его (Ctrl+C там) или: pkill -f 'uvicorn service.app'")
        return
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:app", "--host", "127.0.0.1", "--port", str(PORT)], cwd=ROOT)
    tunnel = subprocess.Popen([CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    if not url:
        print("cloudflared не дал адрес"); _stop(api, tunnel); return
    print(f"\nсервис доступен по адресу {url}")
    _wait_dns(url)
    _publish(url)
    try:
        while api.poll() is None and tunnel.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _stop(api, tunnel)


def _wait_dns(url: str, timeout: int = 180) -> None:
    """Имя туннеля новое: пока его не видит DNS этого компьютера, страница покажет «офлайн».
    Ждём, чтобы не опубликовать адрес раньше, чем он заработает."""
    import socket
    host = url.removeprefix("https://")
    started = time.time()
    while time.time() - started < timeout:
        try:
            socket.getaddrinfo(host, 443)
            print("адрес виден в DNS")
            return
        except socket.gaierror:
            time.sleep(5)
    print("DNS на этом компьютере пока не видит адрес туннеля (роутер отстаёт). Публикую всё равно: "
          "у других он обычно уже работает; здесь помогает DNS 1.1.1.1 в настройках сети или подождать пару минут.")


def _port_busy() -> bool:
    import socket
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def _publish(url: str) -> None:
    CONFIG.write_text(f'// Адрес бэкенда; обновляется tools/serve.py при каждом запуске.\nwindow.LEGOBOT_API = "{url}";\n')
    token = next((l.split("=", 1)[1].strip().strip('"') for l in (ROOT / ".env").read_text().splitlines() if l.startswith("HF_TOKEN_WRITE=")), None)
    if not token:
        print("нет HF_TOKEN_WRITE в .env — config.js на Space не обновлён"); return
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(path_or_fileobj=str(CONFIG), path_in_repo="config.js", repo_id=SPACE, repo_type="space",
                                   commit_message=f"backend {url}")
    print(f"страница https://{SPACE.replace('/', '-')}.static.hf.space/ теперь смотрит на {url}")


def _stop(*procs) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    main()
