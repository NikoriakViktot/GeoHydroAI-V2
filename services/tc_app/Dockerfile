FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libexpat1 curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir terracotta==0.8.2 gunicorn gevent python-dotenv>=1.0.0

WORKDIR /app

# УВАГА: контекст = ./services/terracotta, тому копіюємо все з поточної теки
COPY . /app/services/terracotta/

# дозвіл на entrypoint і створення папок
RUN chmod +x /app/services/terracotta/entrypoint.sh && \
    mkdir -p /app/data/cogs /app/tc_db

ENV PYTHONPATH=/app/services \
    COG_DATA_DIR=/app/data/cogs \
    TC_DB=/app/tc_db/terracotta.sqlite \
    TC_PORT=5000 \
    TC_HOST=0.0.0.0

EXPOSE 5000
ENTRYPOINT ["/app/services/terracotta/entrypoint.sh"]
