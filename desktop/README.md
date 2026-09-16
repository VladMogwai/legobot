# legobot — десктоп (Electron)

Приложение запускает локальный бэкенд (`service/run_server.py`, упакованный PyInstaller'ом)
и показывает его страницу в окне. Всё на этой машине: без сервера, туннелей и интернета
(кроме объёмных фигурок — им нужен HF Space). Кнопка «Открыть в Studio» работает.

Сборка (macOS, из корня репозитория):
```bash
mkdir -p desktop/rembg_home/models/u2net && cp ~/.rembg/models/u2net/u2net.onnx desktop/rembg_home/models/u2net/
.venv/bin/pyinstaller --noconfirm --clean desktop/backend.spec        # -> dist/legobot-server (~620 МБ)
cd desktop && npm install && npx electron-builder --mac dir            # -> desktop/out/mac-arm64/legobot.app
```
Приложение не подписано: при первом запуске — правый клик → Open. Разработка без сборки:
`cd desktop && npm start` (бэкенд берётся из `dist/legobot-server`, а если его нет — из `.venv`).
Результаты: `~/Library/Application Support/legobot/jobs/`.
