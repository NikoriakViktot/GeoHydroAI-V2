FROM postgis/postgis:16-3.5

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    wget lsb-release ca-certificates gnupg

RUN wget --quiet https://packages.apache.org/artifactory/arrow/debian/apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && dpkg -i apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
 && rm apache-arrow-apt-source-latest-$(lsb_release -cs).deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    libarrow-dev libparquet-dev git build-essential cmake \
    postgresql-server-dev-16 \
    postgresql-16-pgvector \
    postgresql-16-ogr-fdw \
    gdal-bin \
    python3-gdal

# parquet_fdw build
RUN git clone https://github.com/adjust/parquet_fdw.git \
 && cd parquet_fdw \
 && make USE_PGXS=1 \
 && make USE_PGXS=1 install \
 && cd .. && rm -rf parquet_fdw

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
