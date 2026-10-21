"""Staging nginx config from the production one (Hetzner staging, 2026-10-21).

    python3 scripts/ops/staging_nginx.py [--host staging.geohydroai.org] [--root /home/ubuntu/geoai]

Writes services/nginx-staging/ next to services/nginx/: the staging host name and its Let's Encrypt certificate,
without the python-course server blocks (Python_curse stays on AWS). docker-compose.staging.yml mounts it.
"""
import argparse
import pathlib
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="staging.geohydroai.org")
ap.add_argument("--root", default=".")
a = ap.parse_args()
src = pathlib.Path(a.root) / "services" / "nginx"
dst = pathlib.Path(a.root) / "services" / "nginx-staging"
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.name != "nginx.conf":
        shutil.copy2(f, dst / f.name)
conf = (src / "nginx.conf").read_text()
marker = "##################################################\n# PYTHON COURSE HTTP"
if marker in conf:
    conf = conf[: conf.index(marker)].rstrip() + "\n}\n"
conf = conf.replace("server_name geohydroai.org www.geohydroai.org;", f"server_name {a.host};")
conf = conf.replace("/etc/letsencrypt/live/www.geohydroai.org/", f"/etc/letsencrypt/live/{a.host}/")
if "python-course" in conf or "server_name geohydroai.org" in conf:
    raise SystemExit("staging_nginx: production host names left in the config; update this script")
(dst / "nginx.conf").write_text(conf)
http = (src / "nginx.http.conf").read_text().replace("server_name geohydroai.org www.geohydroai.org;", f"server_name {a.host};")
(dst / "nginx.http.conf").write_text(http)
print("staging nginx:", re.findall(r"server_name [^;]+;", conf))
