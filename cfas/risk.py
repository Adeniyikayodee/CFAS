"""Satellite layers folded into one flood-risk score.

    R(x) = a*P + b*V + g * theta * mu(theta)

The shape of this equation, the saturation multiplier, and the three plain bands
are the CFAS design (Adeniyi, CFAS, 2026). Each input rides on published work,
cited beside the function that applies it:

    P      rainfall forecast     Google Weather API forward forecast, up to 10 days
                                  (Google Maps Platform, 2026), with the Earth
                                  Engine GFS forecast and CHIRPS as fallbacks (Funk
                                  et al., Sci. Data 2:150066, 2015).
    V      terrain + settlement  TESSERA embeddings (Feng et al., arXiv:2506.20380,
                                  2025; CVPR 2026), the open successor to AlphaEarth
                                  (Brown et al., arXiv:2507.22291, 2025).
    theta  surface soil moisture NASA SMAP (Entekhabi et al., Proc. IEEE
                                  98(5):704, 2010).
    mu     saturation multiplier lifts theta once the ground already holds water.

The score settles into three bands a household can act on: LOW, MEDIUM, HIGH.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GFS = "NOAA/GFS0P25"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
SMAP = "NASA/SMAP/SPL4SMGP/007"
BANDS = (("LOW", 0.34), ("MEDIUM", 0.67))
RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
RAIN_FULL_MM = 50.0   # rainfall mapping to P = 1.0
SOIL_FULL = 0.50      # m3/m3 mapping to theta = 1.0

# Rainfall-driven escalation. Operational flood warning runs a rain-threshold
# trigger alongside any composite index, so a strong forecast can raise the alarm
# on its own even where the standing vulnerability and soil terms are low. P is in
# [0, 1] against RAIN_FULL_MM, so P_MEDIUM=0.6 is ~30 mm and P_HIGH=0.9 is ~45 mm.
# These are deliberately conservative starting points to be calibrated per region.
P_TRIGGER_MEDIUM = 0.60
P_TRIGGER_HIGH = 0.90


@dataclass(frozen=True)
class Risk:
    score: float
    band: str
    rain_mm: float
    soil: float
    vuln: float
    p: float
    theta: float


# --- Earth Engine helpers ---------------------------------------------------
# Every layer below is read on Google Earth Engine, so the heavy lifting stays on
# Google's side and the edge box handles only the final small numbers.
def init_ee(project: str | None):
    import ee
    ee.Initialize(project=project) if project else ee.Initialize()
    return ee


def geocode(country, county, subcounty, community, fallback=(7.80, 6.74)):
    # Names to coordinates through OpenStreetMap's Nominatim. We try the finest
    # place first and widen the query as needed, keeping a sensible local fallback.
    from geopy.geocoders import Nominatim
    geo = Nominatim(user_agent="cfas/1.0")
    for query in (f"{community}, {subcounty}, {county}, {country}",
                  f"{subcounty}, {county}, {country}",
                  f"{county}, {country}"):
        loc = geo.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    return fallback


def aoi_of(ee, lat, lon, km):
    return ee.Geometry.Point([lon, lat]).buffer(km * 1000.0)


def _mean(ee, img, aoi, scale, band, default=0.0):
    stat = img.reduceRegion(ee.Reducer.mean(), aoi, scale,
                            maxPixels=int(1e9), bestEffort=True).get(band)
    val = ee.Number(stat).getInfo()
    return float(val) if val is not None else default


# --- P: rainfall forecast ---------------------------------------------------
# Rainfall is the trigger we watch most closely, so we read a short-range forecast
# from NOAA's Global Forecast System (NCEP GFS) and earn a few hours of lead. When
# forecast frames are sparse, CHIRPS supplies the observed rainfall instead (Funk
# et al., "The climate hazards infrared precipitation with stations," Sci. Data
# 2:150066, 2015). We normalise to [0, 1], with 50 mm over the window reading as 1.0.
def rainfall_p(ee, aoi, start, end, scale):
    gfs = (ee.ImageCollection(GFS).filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lte("forecast_hours", 72))
           .select("total_precipitation_surface"))
    chirps = (ee.ImageCollection(CHIRPS).filterBounds(aoi).filterDate(start, end)
              .select("precipitation").sum())
    img = ee.Image(ee.Algorithms.If(gfs.size().gt(0), gfs.max(), chirps)).rename("rain").unmask(0)
    mm = _mean(ee, img, aoi, scale, "rain")
    return min(mm / RAIN_FULL_MM, 1.0), mm


# --- theta: soil moisture ---------------------------------------------------
# Ground that already holds water turns the next storm dangerous, so we read
# surface soil moisture from NASA SMAP (Entekhabi et al., "The SMAP Mission,"
# Proc. IEEE 98(5):704, 2010). Here 0.5 m3/m3 reads as fully wet (theta = 1.0).
def soil_theta(ee, aoi, start, end, scale):
    sm = ee.ImageCollection(SMAP).filterBounds(aoi).filterDate(start, end).select("sm_surface")
    img = ee.Image(ee.Algorithms.If(sm.size().gt(0), sm.mean(), ee.Image(0.2))).rename("sm").unmask(0.2)
    s = _mean(ee, img, aoi, scale, "sm", default=0.2)
    return min(s / SOIL_FULL, 1.0), s


# --- V: vulnerability from TESSERA -----------------------------------------
# Vulnerability is read in two steps so the slow, network-bound tile fetch can be
# cached for offline use while the cheap embedding-to-score map stays pure. V is
# the slow picture: terrain, watercourses, settlement, the part of risk that
# shapes where water gathers. TESSERA supplies it as an open, global, annual
# embedding at 10 m (Feng et al., "TESSERA: Temporal Embeddings of Surface
# Spectra," arXiv:2506.20380, 2025; served through the GeoTessera library). It
# opens up the ground that AlphaEarth first mapped (Brown et al., "AlphaEarth
# Foundations," arXiv:2507.22291, 2025).
def fetch_embedding(lat, lon, year):
    """Mean 128-d TESSERA embedding for this tile, or None when unavailable.

    This is the only network step in V, so the snapshot caches its result and the
    offline path skips it entirely (see cfas/snapshot.py).
    """
    try:
        from geotessera import GeoTessera
        emb, _, _ = GeoTessera().fetch_embedding(lon=lon, lat=lat, year=year)  # (H, W, 128)
        return emb.reshape(-1, emb.shape[-1]).astype("float32").mean(0)        # (128,)
    except Exception:
        return None


def v_from_embedding(vec, head=None, default=0.5):
    """Map a 128-d embedding to V in [0, 1]. Pure; no network.

    Pass `head` as a (129,) array, 128 weights and a bias, from a trained linear
    probe for a calibrated score. The transparent proxy below keeps the pipeline
    live and honest until that probe arrives. A missing embedding reads as the
    neutral default rather than crashing the score.
    """
    if vec is None:
        return default
    vec = np.asarray(vec, "float32")
    if head is not None:
        w, b = np.asarray(head[:-1], "float32"), float(head[-1])
        return float(1.0 / (1.0 + np.exp(-(vec @ w + b))))
    z = (vec - vec.mean()) / (vec.std() + 1e-6)
    return float(1.0 / (1.0 + np.exp(-(np.linalg.norm(z) / np.sqrt(vec.size) - 1.0))))


def vulnerability_v(lat, lon, year, head=None, default=0.5):
    """Fetch the tile embedding and map it to V. The online convenience wrapper."""
    return v_from_embedding(fetch_embedding(lat, lon, year), head=head, default=default)


def mu(theta, t0=0.6, k=1.0):
    # The saturation multiplier is the CFAS step that lifts the soil-moisture term
    # once the ground is already heavy with water (Adeniyi, CFAS, 2026).
    return 1.0 + k * min(max((theta - t0) / (1.0 - t0), 0.0), 1.0)


def band_of(score):
    # Three plain bands keep the message actionable, so a household reaches for a
    # decision rather than a probability (CFAS design, Adeniyi 2026).
    for name, hi in BANDS:
        if score < hi:
            return name
    return "HIGH"


def fuse(p, v, theta, *, weights, rain_mm, soil,
         trigger_medium=P_TRIGGER_MEDIUM, trigger_high=P_TRIGGER_HIGH) -> Risk:
    """Fold the three normalised terms into one banded Risk. Pure; no network.

    R = a*P + b*V + g*theta*mu(theta): the CFAS fusion that folds forecast,
    vulnerability and wetness into one score (Adeniyi, CFAS, 2026). This is the
    single source of truth for banding, so a live read and a replay from the
    offline snapshot produce the same band for the same inputs.
    """
    a, b, g = weights
    raw = a * p + b * v + g * theta * mu(theta)
    score = min(max(raw / (a + b + g * 2.0), 0.0), 1.0)
    band = band_of(score)
    # Rainfall floor: a strong rainfall forecast escalates the band on its own, so
    # an extreme forecast over dry, low-vulnerability ground still warns. We lift
    # both the band and the score to the band's lower edge, so the operator never
    # sees a HIGH day paired with a contradictory low score.
    if p >= trigger_high:
        band = "HIGH"
        score = max(score, BANDS[1][1])          # >= MEDIUM/HIGH cutoff (0.67)
    elif p >= trigger_medium and RANK[band] < RANK["MEDIUM"]:
        band = "MEDIUM"
        score = max(score, BANDS[0][1])          # >= LOW/MEDIUM cutoff (0.34)
    return Risk(score=score, band=band, rain_mm=round(rain_mm, 1),
                soil=round(soil, 3), vuln=round(v, 3), p=round(p, 3), theta=round(theta, 3))


def assess(ee, aoi, lat, lon, start, end, *, weights, tessera_year, scale, head=None,
           forecast_p=None, trigger_medium=P_TRIGGER_MEDIUM, trigger_high=P_TRIGGER_HIGH) -> Risk:
    # Gather the layers online, then hand them to fuse(). P looks ahead: when a
    # Google Weather forecast is supplied for this day we use it; otherwise we read
    # the Earth Engine GFS forecast, with CHIRPS as fallback.
    if forecast_p is not None:
        p, rain_mm = forecast_p["P"], forecast_p["qpf_mm"]
    else:
        p, rain_mm = rainfall_p(ee, aoi, start, end, scale)
    theta, soil = soil_theta(ee, aoi, start, end, scale)
    v = vulnerability_v(lat, lon, tessera_year, head=head)
    return fuse(p, v, theta, weights=weights, rain_mm=rain_mm, soil=soil,
                trigger_medium=trigger_medium, trigger_high=trigger_high)
