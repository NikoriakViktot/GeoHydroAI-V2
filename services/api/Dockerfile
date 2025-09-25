# services/api/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# системні залежності (варіант з системним GDAL; якщо хочеш wheels — скажу як спростити)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl \
    gdal-bin libgdal-dev \
    proj-bin libproj-dev \
    libpq-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# варіант B: збираємо rasterio/fiona проти системного GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# спочатку numpy (щоб пришвидшити збірки), потім геостек і решта
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "numpy>=1.24,<3" && \
    pip install --no-cache-dir --no-binary rasterio,fiona \
        "rasterio>=1.4.0" "fiona>=1.9.0" && \
    pip install --no-cache-dir -r requirements.txt

# wait-for-it
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# код
COPY . /app

ENV PYTHONPATH=/app

# ❗ CMD у одному рядку (щоб парсер точно не спіткнувся)
CMD ["/wait-for-it.sh", "postgis:5432", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
