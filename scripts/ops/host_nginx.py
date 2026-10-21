"""nginx config for a HOST nginx (systemd, not Docker) from the docker one in services/nginx (owner, 2026-10-21).

    python3 scripts/ops/host_nginx.py --host staging.geohydroai.org [--root /home/ubuntu/geoai]

Writes services/nginx-host/ (git-ignored): nginx.conf + the files it includes. On the server /etc/nginx/nginx.conf
is one line, `include /home/ubuntu/geoai/services/nginx-host/nginx.conf;`, and a systemd path unit reloads nginx when
this file changes (no sudo for the deploy runner).

A host nginx cannot resolve container names, so every upstream becomes a loopback port that the compose files publish:
platform-api 8001, hydro-api 8010, hydro-tiles 8030, terracotta 8050, catalog-api 8095, geoai-ops 8501.
Secrets stay where the host keeps them (/etc/nginx/secrets/{catalog,ops}, root-managed, never in git).
The python-course server blocks are dropped (Python_curse stays on AWS).
"""
import argparse
import pathlib
import re
import shutil

UPSTREAMS = {
    "http://platform-api:8000": "http://127.0.0.1:8001",
    "http://hydro-api:8000": "http://127.0.0.1:8010",
    "http://hydro-tiles:3000": "http://127.0.0.1:8030",
    "http://catalog-api:8095": "http://127.0.0.1:8095",
    "http://geoai-ops:8501": "http://127.0.0.1:8501",
    "server terracotta:5000 resolve;": "server 127.0.0.1:8050;",
}

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, help="geohydroai.org = the production names and certificate as in the docker config; "
                                              "any other name (staging.geohydroai.org) = that name and its own certificate")
ap.add_argument("--root", default=".")
a = ap.parse_args()
root = pathlib.Path(a.root)
src, dst = root / "services" / "nginx", root / "services" / "nginx-host"
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix == ".conf" and f.name not in ("nginx.conf", "nginx.http.conf"):
        shutil.copy2(f, dst / f.name)

conf = (src / "nginx.conf").read_text()
marker = "##################################################\n# PYTHON COURSE HTTP"
if marker in conf:
    conf = conf[: conf.index(marker)].rstrip() + "\n}\n"
PRODUCTION = a.host in ("geohydroai.org", "www.geohydroai.org")
if not PRODUCTION:   # a staging host: its own name and certificate
    conf = conf.replace("server_name geohydroai.org www.geohydroai.org;", f"server_name {a.host};")
    conf = conf.replace("/etc/letsencrypt/live/www.geohydroai.org/", f"/etc/letsencrypt/live/{a.host}/")
for k, v in UPSTREAMS.items():
    conf = conf.replace(k, v)
conf = re.sub(r"\n\s*resolver 127\.0\.0\.11[^;]*;", "\n", conf)          # Docker's DNS: not on the host
conf = conf.replace("/etc/nginx/geoai/", f"{dst.resolve()}/")
conf = "user www-data;\npid /run/nginx.pid;\n\n" + conf

code = "\n".join(line.split("#", 1)[0] for line in conf.splitlines())   # comments may name the containers
left = [h for h in ("platform-api:", "hydro-api:", "hydro-tiles:", "catalog-api:", "geoai-ops:", "terracotta:",
                    "127.0.0.11", "python-course") if h in code]
if not PRODUCTION and "server_name geohydroai.org" in code:
    left.append("server_name geohydroai.org")
if left:
    raise SystemExit(f"host_nginx: docker names left in the config ({left}); update UPSTREAMS")
(dst / "nginx.conf").write_text(conf)
print("host nginx:", dst / "nginx.conf", re.findall(r"server_name [^;]+;", conf))
