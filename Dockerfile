FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY legobot legobot
COPY catalog catalog
COPY vendor/ldraw vendor/ldraw
COPY service service
COPY legobot.toml .
# модель rembg (u2net, ~170 МБ) — скачиваем при сборке, чтобы первый запрос не ждал
RUN python -c "import rembg; rembg.new_session('u2net')"
ENV U2NET_HOME=/root/.u2net
EXPOSE 7860
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
