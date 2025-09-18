FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gdal-bin libgdal-dev libproj-dev libpq-dev curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements лежить у services/api
COPY services/api/requirements.txt ./services/api/requirements.txt
RUN pip install --no-cache-dir -r services/api/requirements.txt

ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# ВАЖЛИВО: копіюємо увесь код, який імпортується
COPY core ./core
COPY infrastructure ./infrastructure
COPY services/api ./services/api

# зробимо пакети видимими
ENV PYTHONPATH=/app

# чекати Postgres у compose-сервісі "postgis"
CMD ["/wait-for-it.sh", "postgis:5432", "--", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
