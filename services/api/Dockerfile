FROM python:3.11-slim

WORKDIR /code

#RUN apt-get update && apt-get install -y --no-install-recommends \
#    gdal-bin \
#    libgdal-dev \
#    libproj-dev \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка всех необходимых системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libpq-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

COPY . .

CMD ["/wait-for-it.sh", "db:5432", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
