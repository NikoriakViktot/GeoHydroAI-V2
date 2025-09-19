# debug_utils.py
from __future__ import annotations
import time, json, logging, functools, traceback
log = logging.getLogger("debug")

def to_jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)

def trace(name=None):
    """Логує виклики: args/kwargs, час, результат|помилку."""
    def deco(fn):
        nm = name or fn.__name__
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            t0 = time.perf_counter()
            log.info("→ %s args=%s kwargs=%s", nm, to_jsonable(args), to_jsonable(kwargs))
            try:
                res = fn(*args, **kwargs)
                dt = (time.perf_counter() - t0)*1000
                log.info("← %s ok in %.1f ms result=%s", nm, dt, to_jsonable(res if isinstance(res,(str,int,float,bool)) else type(res).__name__))
                return res
            except Exception as e:
                dt = (time.perf_counter() - t0)*1000
                tb = traceback.format_exc(limit=3)
                log.exception("← %s FAIL in %.1f ms: %s\n%s", nm, dt, e, tb)
                raise
        return wrap
    return deco
