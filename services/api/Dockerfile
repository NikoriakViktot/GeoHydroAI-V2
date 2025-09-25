# services/api/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# 1) Системні залежності + GDAL/PROJ з хедерами
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl \
    gdal-bin libgdal-dev \
    proj-bin libproj-dev \
    libpq-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Підказки для збірки/рантайму (щоб rasterio/fiona знайшли системний GDAL/PROJ)
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# 3) Python deps: спочатку numpy, далі геостек, ЗІБРАТИ проти системного GDAL
COPY services/api/requirements.txt ./services/api/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "numpy>=1.24,<3" && \
    pip install --no-cache-dir --no-binary rasterio,fiona \
        "rasterio>=1.4.0" "fiona>=1.9.0" && \
    pip install --no-cache-dir -r services/api/requirements.txt

# 4) wait-for-it
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# 5) Код
COPY services/api ./services/api
ENV PYTHONPATH=/app

# 6) Старт
CMD ["/wait-for-it.sh", "postgis:5432", "--",
     "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
