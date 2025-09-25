# core/usecases/compare_dem.py
from typing import Dict, Iterable, List, Optional

class CompareDemUseCase:
    """
    Онлайн use-case: бере точки ICESat-2 з PostGIS-репозиторію та семплить значення шарів через Terracotta.
    Повертає агрегати (MAE/RMSE/Bias) і рядки-спостереження (для карт, якщо треба).
    """

    def __init__(self, repo_points, sampler, slope_layer: Optional[str], hand_layer: Optional[str]):
        self.dem_repo = repo_points
        self.sampler = sampler
        self.slope_layer = slope_layer
        self.hand_layer = hand_layer

    def execute(self, aoi_wkt: str, dem_names: Iterable[str], limit: int = 200000):
        metrics: Dict[str, Dict[str, float]] = {
            d: {"n": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0} for d in dem_names
        }
        rows: List[Dict] = []

        for pid, h, lon, lat in self.dem_repo.icesat_points_in_wkt(aoi_wkt, limit):
            slope = self.sampler.sample(self.slope_layer, lon, lat) if self.slope_layer else None
            hand  = self.sampler.sample(self.hand_layer,  lon, lat) if self.hand_layer  else None

            for dem in dem_names:
                v = self.sampler.sample(dem, lon, lat)
                if v is None:
                    continue
                diff = h - v
                rows.append({
                    "icesat_id": pid, "dem_name": dem, "dem_value": v,
                    "diff": diff, "slope": slope, "hand": hand,
                    "lon": lon, "lat": lat
                })
                m = metrics[dem]
                m["n"] += 1
                m["mae"]  += abs(diff)
                m["rmse"] += diff * diff
                m["bias"] += diff

        for dem, m in metrics.items():
            n = max(1, m["n"])
            m["mae"]  = m["mae"] / n
            m["rmse"] = (m["rmse"] / n) ** 0.5
            m["bias"] = m["bias"] / n

        return {"metrics": metrics, "rows": rows}
