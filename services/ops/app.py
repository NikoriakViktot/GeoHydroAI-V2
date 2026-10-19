"""GeoHydroAI ops dashboard (owner, 2026-10-19: «логування задач на сервері і роботи воркерів … стрімліт дашборд …
по можливості запускати воркери»).

What it reads, all on this host:
  * Docker: every container's state, health, restarts, memory, and its log (Docker API over the mounted socket);
  * systemd: the units of WATCHED_UNITS (state, restarts, memory) and their journal (host journal, read-only);
  * ERA5-Land: the worker's progress, counted from its journal lines (raw months from EDH / CDS, hourly, daily).
What it may do: start / stop / restart only the containers and units in CONTROL (never the database, nginx or the
certificates). Every action is logged to /data/actions.log and the page.

Behind nginx /ops/ with HTTP basic auth (the password file lives outside git). Streamlit runs with baseUrlPath=ops.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

import docker
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GeoHydroAI ops", page_icon="🛠", layout="wide")

JOURNAL = os.environ.get("OPS_JOURNAL_DIR", "/host/journal")
ACTIONS = Path(os.environ.get("OPS_DATA", "/data")) / "actions.log"
WATCHED_UNITS = [u for u in os.environ.get("OPS_UNITS", "era5land-worker.service").split(",") if u]
# what the page may start / stop / restart: workers and services that recover on their own; never postgis, nginx, certbot
CONTROL_CONTAINERS = set(os.environ.get(
    "OPS_CONTROL_CONTAINERS",
    "platform-worker,server-geoproc-worker-1,swot-worker,hydro-admin,hydro-api,platform-api,server-geoproc-1,"
    "geoai-terracotta-1,hydro-tiles,catalog-api,synop-fetch").split(","))
CONTROL_UNITS = set(os.environ.get("OPS_CONTROL_UNITS", "era5land-worker.service").split(","))
ERR = re.compile(r"error|exception|traceback|failed|oom|killed|critical|timeout", re.I)
SYSTEMD_ENV = {**os.environ, "SYSTEMD_PAGER": "", "SYSTEMD_COLORS": "0"}


@st.cache_resource
def dk():
    return docker.from_env()


def sh(*cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=SYSTEMD_ENV)
    return (r.stdout or "") + (r.stderr if r.returncode else "")


def journal(unit: str | None, lines: int, since: str | None = None) -> str:
    cmd = ["journalctl", "-D", JOURNAL, "--no-pager", "-o", "short-iso", "-n", str(lines)]
    if unit:
        cmd += ["-u", unit]
    if since:
        cmd += ["--since", since]
    return sh(*cmd, timeout=60)


def _unit_path(unit: str) -> str:
    """systemd's D-Bus object path of a unit: every byte that is not [A-Za-z0-9] becomes _XX"""
    return "/org/freedesktop/systemd1/unit/" + "".join(c if c.isalnum() else f"_{ord(c):02x}" for c in unit)


def _prop(unit: str, iface: str, name: str):
    out = sh("busctl", "--system", "get-property", "org.freedesktop.systemd1", _unit_path(unit), f"org.freedesktop.systemd1.{iface}", name)
    parts = out.strip().split(" ", 1)
    if len(parts) < 2:
        return None
    t, v = parts
    if t == "s":                                     # busctl prints strings C-escaped (UTF-8 as \ooo octets)
        return v.strip('"').encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    return int(v) if t in ("u", "t", "i", "x") and v.lstrip("-").isdigit() else v


def unit_props(unit: str) -> dict:
    """state, restarts and memory of a host unit over the system D-Bus (systemctl does not run inside a container)"""
    mb = lambda v: round(v / 2**20) if isinstance(v, int) and v < 2**62 else None  # noqa: E731
    since = _prop(unit, "Unit", "ActiveEnterTimestamp")
    return {"unit": unit, "state": f"{_prop(unit, 'Unit', 'ActiveState')} / {_prop(unit, 'Unit', 'SubState')}",
            "restarts": _prop(unit, "Service", "NRestarts"), "mem_MB": mb(_prop(unit, "Service", "MemoryCurrent")),
            "peak_MB": mb(_prop(unit, "Service", "MemoryPeak")),
            "since": dt.datetime.fromtimestamp(since / 1e6, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if isinstance(since, int) and since else None,
            "what": _prop(unit, "Unit", "Description")}


def unit_action(unit: str, action: str) -> str:
    method = {"start": "StartUnit", "stop": "StopUnit", "restart": "RestartUnit"}[action]
    return sh("busctl", "--system", "call", "org.freedesktop.systemd1", "/org/freedesktop/systemd1",
              "org.freedesktop.systemd1.Manager", method, "ss", unit, "replace")


def record(action: str, target: str, result: str):
    ACTIONS.parent.mkdir(parents=True, exist_ok=True)
    who = st.context.headers.get("X-Remote-User") or "basic-auth user"
    with ACTIONS.open("a") as f:
        f.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "who": who,
                            "action": action, "target": target, "result": result[:300]}, ensure_ascii=False) + "\n")


def host_line() -> str:
    try:
        mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in Path("/host/proc/meminfo").read_text().splitlines())}
        load = Path("/host/proc/loadavg").read_text().split()[:3]
        st_ = os.statvfs("/data")
        free_gb = st_.f_bavail * st_.f_frsize / 1e9
        used = 1 - st_.f_bavail / st_.f_blocks
        return (f"RAM available **{mem['MemAvailable'] / 2**20:.1f}** of {mem['MemTotal'] / 2**20:.1f} GB · "
                f"swap used {(mem['SwapTotal'] - mem['SwapFree']) / 2**20:.1f} GB · load {' '.join(load)} · "
                f"disk **{used:.0%}** used, {free_gb:.0f} GB free")
    except Exception as e:  # noqa: BLE001
        return f"host metrics unavailable: {e}"


st.title("GeoHydroAI — сервер, воркери, логи")
st.caption(host_line() + f" · {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
tab_over, tab_logs, tab_era5, tab_ctl = st.tabs(["Огляд", "Логи", "ERA5-Land", "Керування"])

# ---------------------------------------------------------------- overview
with tab_over:
    @st.fragment(run_every=30)
    def overview():
        rows = []
        for c in dk().containers.list(all=True):
            s = c.attrs["State"]
            rows.append({"контейнер": c.name, "стан": s.get("Status"), "код": s.get("ExitCode"), "здоров'я": (s.get("Health") or {}).get("Status", "—"),
                         "рестартів": c.attrs.get("RestartCount", 0), "старт": (s.get("StartedAt") or "")[:19].replace("T", " "),
                         "образ": c.attrs.get("Config", {}).get("Image", "?"), "OOM": s.get("OOMKilled")})
        df = pd.DataFrame(rows).sort_values(["стан", "контейнер"])
        # a one-shot job that exited 0 is fine; a crash, a restart loop, unhealthy or OOM is not
        bad = df[((df["стан"] == "exited") & (df["код"] != 0)) | df["стан"].isin(["restarting", "dead"])
                 | (df["здоров'я"] == "unhealthy") | df["OOM"]]
        if len(bad):
            st.warning("Потребують уваги: " + ", ".join(bad["контейнер"]))
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.subheader("Сервіси systemd")
        st.dataframe(pd.DataFrame([unit_props(u) for u in WATCHED_UNITS]), hide_index=True, use_container_width=True)
        if st.toggle("Пам'ять і CPU контейнерів (займає кілька секунд)", key="stats"):
            srows = []
            for c in dk().containers.list():
                s = c.stats(stream=False)
                mem = s.get("memory_stats", {})
                cpu, pre = s.get("cpu_stats", {}), s.get("precpu_stats", {})
                d_cpu = cpu.get("cpu_usage", {}).get("total_usage", 0) - pre.get("cpu_usage", {}).get("total_usage", 0)
                d_sys = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
                srows.append({"контейнер": c.name, "пам'ять, МБ": round(mem.get("usage", 0) / 2**20),
                              "ліміт, МБ": round(mem.get("limit", 0) / 2**20),
                              "CPU, %": round(100 * d_cpu / d_sys * cpu.get("online_cpus", 1), 1) if d_sys > 0 else 0})
            st.dataframe(pd.DataFrame(srows).sort_values("пам'ять, МБ", ascending=False), hide_index=True, use_container_width=True)
    overview()

# ---------------------------------------------------------------- logs
with tab_logs:
    names = sorted(c.name for c in dk().containers.list(all=True))
    sources = [f"unit: {u}" for u in WATCHED_UNITS] + [f"docker: {n}" for n in names] + ["journal: весь хост"]
    c1, c2, c3, c4 = st.columns([3, 1, 2, 1])
    src = c1.selectbox("Джерело", sources, key="src")
    n = c2.number_input("Рядків", 50, 5000, 300, 50, key="n")
    flt = c3.text_input("Фільтр (текст або regex)", key="flt")
    only_err = c4.toggle("Лише помилки", key="err")
    live = st.toggle("Оновлювати кожні 10 с", key="live")

    @st.fragment(run_every=10 if live else None)
    def show_logs():
        kind, _, name = src.partition(": ")
        if kind == "docker":
            text = dk().containers.get(name).logs(tail=int(n), timestamps=True).decode("utf-8", "replace")
        elif kind == "unit":
            text = journal(name, int(n))
        else:
            text = journal(None, int(n))
        lines = text.splitlines()
        if flt:
            try:
                rx = re.compile(flt, re.I)
                lines = [l for l in lines if rx.search(l)]
            except re.error:
                lines = [l for l in lines if flt.lower() in l.lower()]
        if only_err:
            lines = [l for l in lines if ERR.search(l)]
        st.caption(f"{len(lines)} рядків · {src}")
        st.code("\n".join(lines[-int(n):]) or "(порожньо)", language="log")
    show_logs()

# ---------------------------------------------------------------- ERA5-Land
with tab_era5:
    @st.fragment(run_every=60)
    def era5():
        unit = "era5land-worker.service"
        p = unit_props(unit)
        st.markdown(f"**{p['what']}** — {p['state']}, рестартів {p['restarts']}, пам'ять {p['mem_MB']} МБ (пік {p['peak_MB']}), з {p['since']}")
        text = journal(unit, 200000, since="-30 days")
        ev = []
        for line in text.splitlines():
            m = re.match(r"(\S+)\s+\S+\s+\S+:\s+(.*)$", line)
            if not m:
                continue
            t, msg = m.groups()
            if (r := re.match(r"raw/(\w+)/(\d{4}-\d{2})\.nc: (\d+) MB (from EDH|in S3)", msg)):
                ev.append((t, "raw", r[1], r[2], "EDH" if "EDH" in r[4] else "CDS"))
            elif (r := re.match(r"hourly/(\w+) (\d{4}-\d{2})\.nc", msg)):
                ev.append((t, "hourly", r[1], r[2], ""))
            elif (r := re.match(r"daily/(\w+) (\d{4}-\d{2})\.nc", msg)):
                ev.append((t, "daily", r[1], r[2], ""))
        if not ev:
            st.info("У журналі ще немає готових місяців.")
            return
        df = pd.DataFrame(ev, columns=["time", "stage", "group", "month", "source"])
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
        total_months = (dt.date.today().year - 1989) * 12 + dt.date.today().month + 1
        cols = st.columns(3)
        for i, g in enumerate(["met", "flux", "land"]):
            d = df[df.group == g]
            cnt = {s: d[d.stage == s].month.nunique() for s in ("raw", "hourly", "daily")}
            last_h = d[d.time > d.time.max() - pd.Timedelta(hours=1)] if len(d) else d
            rate = last_h[last_h.stage == "raw"].month.nunique()
            left = max(total_months - cnt["raw"], 0)
            cols[i].metric(f"{g}: сирих місяців", f"{cnt['raw']} / {total_months}",
                           help=f"погодинних {cnt['hourly']}, добових {cnt['daily']}")
            cols[i].caption(f"погодинних {cnt['hourly']} · добових {cnt['daily']} · за останню годину {rate} міс."
                            + (f" · лишилось ~{left / rate:.0f} год" if rate else ""))
        hourly = df.set_index("time").groupby([pd.Grouper(freq="1h"), "stage"]).size().unstack(fill_value=0)
        st.bar_chart(hourly, height=220)
        st.dataframe(df.sort_values("time", ascending=False).head(50), hide_index=True, use_container_width=True)
        errs = [l for l in text.splitlines()[-3000:] if ERR.search(l) and "RuntimeWarning" not in l]
        if errs:
            with st.expander(f"Помилки й попередження ({len(errs)})"):
                st.code("\n".join(errs[-100:]), language="log")
    era5()

# ---------------------------------------------------------------- control
with tab_ctl:
    st.markdown("Дозволено запускати, зупиняти й перезапускати лише воркери та сервіси, що відновлюються самі. "
                "База даних, nginx і сертифікати — тільки перегляд.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Контейнери")
        allowed = sorted(n for n in (c.name for c in dk().containers.list(all=True)) if n in CONTROL_CONTAINERS)
        target = st.selectbox("Контейнер", allowed, key="ct")
        act = st.radio("Дія", ["restart", "start", "stop"], horizontal=True, key="ca")
        sure = st.checkbox(f"Так, {act} {target}", key="cs")
        if st.button("Виконати", key="cb", disabled=not sure, type="primary"):
            try:
                getattr(dk().containers.get(target), act)()
                record(act, target, "ok"); st.success(f"{act} {target}: виконано")
            except Exception as e:  # noqa: BLE001
                record(act, target, repr(e)); st.error(repr(e))
    with c2:
        st.subheader("Сервіси systemd")
        unit = st.selectbox("Сервіс", sorted(CONTROL_UNITS), key="ut")
        uact = st.radio("Дія", ["restart", "start", "stop"], horizontal=True, key="ua")
        usure = st.checkbox(f"Так, {uact} {unit}", key="us")
        if st.button("Виконати", key="ub", disabled=not usure, type="primary"):
            out = unit_action(unit, uact)
            ok = out.strip().startswith("o ")                 # busctl prints the job object path on success
            record(uact, unit, "ok" if ok else out)
            (st.success if ok else st.error)(f"{uact} {unit}: виконано" if ok else out.strip())
    st.subheader("Журнал дій")
    if ACTIONS.exists():
        st.dataframe(pd.DataFrame([json.loads(l) for l in ACTIONS.read_text().splitlines()[-200:]][::-1]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("Дій ще не було.")
