"""Forward rainfall forecast from the Google Weather API.

CFAS predicts, so the rainfall term P looks ahead rather than back. The Google
Maps Platform Weather API returns up to 10 days of daily forecast for a point,
drawing on both AI weather models and numerical prediction (Google Maps Platform,
"Weather API," developers.google.com/maps/documentation/weather, 2026).

We read two fields from each day, the forecast rainfall amount (qpf, in mm) and
its probability (percent), and we read both the daytime and nighttime blocks so
the day total covers a full 24 hours. Heavy rain that is likely should weigh more
than heavy rain at long odds, so P folds amount and probability together, with a
tunable lean between them.

The call is a simple GET:

    https://weather.googleapis.com/v1/forecast/days:lookup
        ?key=KEY&location.latitude=LAT&location.longitude=LON&days=N

Set GOOGLE_WEATHER_KEY in the environment. When the key or the network is away,
the caller falls back to the Earth Engine GFS and CHIRPS path in risk.py, so the
pipeline keeps running.
"""
from __future__ import annotations

from dataclasses import dataclass

WEATHER_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"
RAIN_FULL_MM = 50.0   # forecast rainfall (mm) that maps to a full P of 1.0


@dataclass(frozen=True)
class DayRain:
    date: str          # YYYY-MM-DD
    qpf_mm: float      # forecast rainfall for the day, day + night
    prob: float        # mean chance of rain across the day, 0..1


def _block_qpf(block: dict) -> tuple[float, float]:
    p = (block or {}).get("precipitation", {})
    qpf = float(p.get("qpf", {}).get("quantity", 0.0) or 0.0)
    prob = float(p.get("probability", {}).get("percent", 0.0) or 0.0) / 100.0
    return qpf, prob


def fetch_daily(lat: float, lon: float, key: str, days: int = 7, timeout: int = 30) -> list[DayRain]:
    """Pull the daily forecast and reduce each day to one DayRain. The API caps at
    10 days, so we ask for the smaller of the request and that ceiling."""
    import requests
    out: list[DayRain] = []
    token = None
    want = max(1, min(days, 10))
    while len(out) < want:
        params = {"key": key, "location.latitude": lat, "location.longitude": lon,
                  "days": want, "pageSize": want}
        if token:
            params["pageToken"] = token
        r = requests.get(WEATHER_URL, params=params, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        for d in body.get("forecastDays", []):
            disp = d.get("displayDate", {})
            y, m, day = disp.get("year"), disp.get("month"), disp.get("day")
            if not (y and m and day):
                # A day with no usable date cannot be matched to the run loop, so
                # we surface it rather than emit a phantom key that silently drops.
                raise ValueError(f"forecast day missing a usable displayDate: {disp!r}")
            date = f"{y:04d}-{m:02d}-{day:02d}"
            dq, dpr = _block_qpf(d.get("daytimeForecast", {}))
            nq, npr = _block_qpf(d.get("nighttimeForecast", {}))
            out.append(DayRain(date=date, qpf_mm=dq + nq, prob=max(dpr, npr)))
        token = body.get("nextPageToken")
        if not token:
            break
    return out[:want]


def p_from_day(day: DayRain, prob_lean: float = 0.35) -> float:
    """Map one forecast day to P in [0, 1].

    Amount drives the signal; probability scales it. prob_lean sets how much the
    chance of rain matters: 0.0 trusts the amount alone, 1.0 trusts amount times
    probability. The default keeps most of the amount while easing off rain that
    is forecast but unlikely.
    """
    amount = min(day.qpf_mm / RAIN_FULL_MM, 1.0)
    weight = (1.0 - prob_lean) + prob_lean * day.prob
    return max(0.0, min(amount * weight, 1.0))


def forecast_index(lat: float, lon: float, key: str, *, days: int = 7,
                   prob_lean: float = 0.35) -> dict[str, dict]:
    """Return {date: {"P": float, "qpf_mm": float, "prob": float}} for the window."""
    return {d.date: {"P": p_from_day(d, prob_lean), "qpf_mm": round(d.qpf_mm, 1),
                     "prob": round(d.prob, 2)}
            for d in fetch_daily(lat, lon, key, days=days)}
