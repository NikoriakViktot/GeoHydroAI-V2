# ---------- builder: тільки для збірки parquet_fdw ----------
FROM postgis/postgis:16-3.5 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget lsb-release ca-certificates gnupg git build-essential cmake \
    postgresql-server-dev-16 \
 && rm -rf /var/lib/apt/lists/*

# Arrow APT source
RUN wget --quiet https://packages.apache.org/artifactory/arrow/debian/apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && dpkg -i apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends libarrow-dev libparquet-dev \
 && rm -rf /var/lib/apt/lists/*

# Збираємо parquet_fdw (PIN на стабільний коміт)
ARG PARQUET_FDW_COMMIT=2f7f7f1   # <-- заміни на актуальний SHA/реліз
RUN git clone https://github.com/adjust/parquet_fdw.git /tmp/parquet_fdw \
 && cd /tmp/parquet_fdw \
 && git checkout ${PARQUET_FDW_COMMIT} \
 && make USE_PGXS=1 \
 && make USE_PGXS=1 install

# ---------- final: тільки рантайм ----------
FROM postgis/postgis:16-3.5
ENV DEBIAN_FRONTEND=noninteractive

# Arrow APT source + runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget lsb-release ca-certificates gnupg \
 && wget --quiet https://packages.apache.org/artifactory/arrow/debian/apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && dpkg -i apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && apt-get update && apt-get install -y --no-install-recommends \
    # runtime, не dev:
    libarrow-glib-dev libarrow-dev libparquet-dev \
    postgresql-16-pgvector \
    postgresql-16-ogr-fdw \
    gdal-bin python3-gdal \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Копіюємо лише зібраний FDW
COPY --from=builder /usr/lib/postgresql/16/lib/parquet_fdw.so /usr/lib/postgresql/16/lib/parquet_fdw.so
COPY --from=builder /usr/share/postgresql/16/extension/parquet_fdw* /usr/share/postgresql/16/extension/

# (опц) локаль/таймзона/encoding
ENV LANG=C.UTF-8

# ENTRYPOINT як у базового образу (postgres)
