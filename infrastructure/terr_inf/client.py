# infrastructure/terracotta/client.py
from typing import Optional
import os, requests

class TerracottaSampler:
    """
    Простий клієнт до Terracotta API (/value).
    """
    def __init__(self, base: Optional[str] = None, timeout: float = 10.0):
        self.base = base or os.getenv("TERRACOTTA_URL", "http://terracotta:5000")
        self.timeout = timeout

    def sample(self, layer: str, lon: float, lat: float) -> Optional[float]:
        r = requests.get(
            f"{self.base}/value",
            params={"layer": layer, "lon": lon, "lat": lat},
            timeout=self.timeout
        )
        if r.status_code != 200:
            return None
        data = r.json()
        v = data.get("value")
        return float(v) if v is not None else None
