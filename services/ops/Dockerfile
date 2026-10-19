# GeoHydroAI ops dashboard (Streamlit). Ubuntu 24.04 like the host, so journalctl reads the host journal format and
# systemctl speaks the host systemd's protocol. Built on the workstation and loaded on the server (README) — nothing
# is built on the production host and no GitHub-hosted minutes are spent.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv systemd ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir "streamlit==1.45.1" "docker==7.1.0" "pandas==2.2.3"
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
COPY .streamlit /app/.streamlit
COPY app.py /app/app.py
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/ops/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py"]
