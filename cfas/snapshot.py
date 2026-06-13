"""Offline-first snapshot of the satellite layers.

The risk inputs reduce to a few small numbers per day plus one 128-d TESSERA
embedding, so the edge box can pull them into a local file while it has a
connection, then compute, draft and broadcast from that file with no network.
This is the data-layer half of running on the device: Cactus keeps the models
local; the snapshot keeps the inputs local.

We store the RAW inputs, not the derived P, V and theta, so the calibration
dials stay tunable offline: alpha/beta/gamma, prob_lean, the rain triggers and a
trained --vuln-head probe all re-band the same snapshot without another fetch.

    cfas-snapshot/1 schema:
      {
        schema, fetched_at,
        location {community, subcounty, county, country, lat, lon},
        tessera  {year, embedding: [128 floats] | null},
        days     {YYYY-MM-DD: {qpf_mm, prob, soil, source}}
      }

A note on freshness: TESSERA is annual, so the embedding ages slowly, but the
rainfall forecast and SMAP soil moisture are time-sensitive. A warning computed
from days-old soil moisture is dangerous, so callers guard on fetched_at with
is_stale() before trusting an old snapshot offline.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import risk as R
from .forecast import DayRain, forecast_index, p_from_day
from .risk import SOIL_FULL

SCHEMA = "cfas-snapshot/1"


def build(s, *, ee_project, weather_key, prob_lean, scale, log=None) -> dict:
    """Pull the window's layers online and return a snapshot dict. Network-bound.

    `s` is the run Settings. The GFS path has no probability, so it is stored as
    prob = 1.0; with prob_lean folded in at score time this reproduces the
    amount-only P the live GFS path would have produced, keeping live and offline
    bands identical.
    """
    say = log.info if log is not None else (lambda *a, **k: None)
    warn = log.warning if log is not None else (lambda *a, **k: None)

    ee = R.init_ee(ee_project)
    lat, lon = R.geocode(s.country, s.county, s.subcounty, s.community)
    aoi = R.aoi_of(ee, lat, lon, s.buffer_km)
    say("AOI %.4f, %.4f  (%g km)", lat, lon, s.buffer_km)

    vec = R.fetch_embedding(lat, lon, s.tessera_year)
    say("TESSERA embedding %s", "loaded" if vec is not None
        else "unavailable; V will fall back to the proxy default")

    forecast = {}
    if weather_key:
        try:
            span = (s.end - s.start).days + 1
            forecast = forecast_index(lat, lon, weather_key, days=span, prob_lean=prob_lean)
            say("Google Weather forecast loaded for %d day(s)", len(forecast))
        except Exception as e:
            warn("Google Weather forecast unavailable (%s); using GFS", e)
    else:
        say("No GOOGLE_WEATHER_KEY set; rainfall will read Earth Engine GFS")

    days = {}
    day = s.start
    while day <= s.end:
        d = day.isoformat()
        nxt = (day + dt.timedelta(days=1)).isoformat()
        if d in forecast:
            qpf, prob, src = forecast[d]["qpf_mm"], forecast[d]["prob"], "google-weather"
        else:
            _, mm = R.rainfall_p(ee, aoi, d, nxt, scale)
            qpf, prob, src = round(mm, 1), 1.0, "gfs/chirps"
        _, soil = R.soil_theta(ee, aoi, d, nxt, scale)
        days[d] = {"qpf_mm": round(float(qpf), 1), "prob": round(float(prob), 2),
                   "soil": round(float(soil), 3), "source": src}
        day += dt.timedelta(days=1)

    return {
        "schema": SCHEMA,
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "location": {"community": s.community, "subcounty": s.subcounty,
                     "county": s.county, "country": s.country, "lat": lat, "lon": lon},
        "tessera": {"year": s.tessera_year,
                    "embedding": vec.tolist() if vec is not None else None},
        "days": days,
    }


def save(path, snap) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2))


def load(path):
    """Read a snapshot, or None when the file is absent. Raises on a wrong schema."""
    path = Path(path)
    if not path.exists():
        return None
    snap = json.loads(path.read_text())
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"unrecognised snapshot schema {snap.get('schema')!r} at {path}")
    return snap


def age_hours(snap, now=None) -> float:
    now = now or dt.datetime.now()
    return (now - dt.datetime.fromisoformat(snap["fetched_at"])).total_seconds() / 3600.0


def is_stale(snap, max_age_hours, now=None) -> bool:
    """True when the time-sensitive layers are older than max_age_hours."""
    return age_hours(snap, now) > max_age_hours


def risk_for_day(snap, day, *, weights, head=None, prob_lean=0.35,
                 trigger_medium=R.P_TRIGGER_MEDIUM, trigger_high=R.P_TRIGGER_HIGH):
    """Recompute a day's Risk from the cached raw inputs. Pure; no network.

    Returns None when the snapshot holds no entry for the day. The derived terms
    (P, theta, V) are rebuilt here and the banding is delegated to risk.fuse(), so
    this shares the live path's one source of truth for the band.
    """
    d = snap.get("days", {}).get(day)
    if d is None:
        return None
    p = p_from_day(DayRain(day, float(d["qpf_mm"]), float(d["prob"])), prob_lean)
    theta = min(float(d["soil"]) / SOIL_FULL, 1.0)
    emb = snap.get("tessera", {}).get("embedding")
    v = R.v_from_embedding(np.asarray(emb, "float32") if emb is not None else None, head=head)
    return R.fuse(p, v, theta, weights=weights, rain_mm=float(d["qpf_mm"]), soil=float(d["soil"]),
                  trigger_medium=trigger_medium, trigger_high=trigger_high)
