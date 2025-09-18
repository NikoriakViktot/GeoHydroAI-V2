# syntax=docker/dockerfile:1.7

FROM mambaorg/micromamba:1.5.8-jammy AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
COPY environment.yml /tmp/environment.yml
RUN micromamba config set channel_priority strict && \
    micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

FROM mambaorg/micromamba:1.5.8-jammy AS dash
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
COPY --from=base /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8050 \
     GUNICORN_CMD_ARGS="--workers=2 --threads=4 --timeout 120" \
    PYTHONPATH=/app
COPY . /app
EXPOSE 8050
CMD ["bash", "-lc", "gunicorn app:server -b 0.0.0.0:${PORT}"]

# (опційно) debug-стейдж, але НЕ фінальний
FROM ubuntu:22.04 AS debug
RUN apt-get update && apt-get install -y curl less
ENTRYPOINT ["/bin/bash"]
