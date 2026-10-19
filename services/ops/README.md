# Ops dashboard (`/ops/`)

Streamlit page for the server: containers and systemd units (state, health, restarts, memory), their logs, the
ERA5-Land worker's progress, and start / stop / restart of the allowed workers (`OPS_CONTROL_*` in `app.py`; never
postgis, nginx or certbot). Every action goes to `/home/ubuntu/geoai-ops/data/actions.log`.

Access: https://geohydroai.org/ops/ behind HTTP basic auth. The password file is outside git:
`/home/ubuntu/geoai-ops/nginx/htpasswd` (mounted into nginx as `/etc/nginx/secrets/ops/htpasswd`). Add or change a user:

    printf 'owner:%s\n' "$(openssl passwd -apr1)" > /home/ubuntu/geoai-ops/nginx/htpasswd

The container mounts the Docker socket and talks to the host systemd, i.e. it can control the host: keep it behind
the password and do not widen the allow-lists without need.

Image (built on the workstation, not on the server, not on GitHub-hosted runners):

    docker build -t geoai-ops:latest services/ops
    docker save geoai-ops:latest | gzip | ssh -4 ubuntu@geohydroai.org 'gunzip | docker load'
    ssh -4 ubuntu@geohydroai.org 'cd ~/geoai && docker compose up -d ops'
