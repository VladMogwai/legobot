# legobot service

FastAPI поверх `legobot`: `POST /jobs` (фото) → `GET /jobs/{id}` → файлы `/jobs/{id}/files/…`.
Docker-образ собирается из `Dockerfile` в корне репозитория (порт 7860 — как ждёт Hugging Face Spaces).

Деплой на HF Space (Docker): создать Space типа Docker, запушить этот репозиторий в его git
(`git remote add space https://huggingface.co/spaces/<user>/legobot && git push space main`).
Локально: `uvicorn service.app:app --port 7860`.
