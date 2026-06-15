"""End-to-end integration: a forecast value flowing through run.py into the band,
the assessment ledger, and the broadcast sheet. Unit tests check each part in
isolation; this checks the wiring between them, which is where wiring bugs hide.

Run with pytest, or directly:  python tests/test_integration.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import risk as R    # noqa: E402
from cfas import run as RUN   # noqa: E402


def _stub_satellite_layers():
    # Hold the satellite layers fixed at dry, low-vulnerability values so the
    # forecast is the only thing that can move the band. These are the network
    # reads the snapshot build performs; stubbing them keeps the test offline.
    R.soil_theta = lambda *a, **k: (0.10, 0.05)
    R.fetch_embedding = lambda *a, **k: None   # V falls back to the proxy default
    R.init_ee = lambda *a, **k: None
    R.geocode = lambda *a, **k: (7.80, 6.74)
    R.aoi_of = lambda *a, **k: None


def test_forecast_drives_pipeline_end_to_end():
    _stub_satellite_layers()
    d0 = dt.date(2026, 6, 3)
    d1 = d0 + dt.timedelta(days=1)
    fake = {d0.isoformat(): {"P": 1.0, "qpf_mm": 60.0, "prob": 0.95},   # torrential
            d1.isoformat(): {"P": 0.05, "qpf_mm": 1.0, "prob": 0.05}}    # dry

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.yaml"
        cfg.write_text(
            "location: {country: Nigeria, county: Kogi, subcounty: Lokoja, "
            "community: Adankolo, landmark: the market bridge}\n"
            f"window: {{start: {d0.isoformat()}, end: {d1.isoformat()}, "
            "expected_hour: this evening}\n"
            "weights: {alpha: 0.40, beta: 0.30, gamma: 0.30}\n"
            "languages: [hausa]\nalert_from: MEDIUM\n")
        out = Path(tmp) / "alerts"
        argv = ["run", "--config", str(cfg), "--outdir", str(out)]
        with patch.object(sys, "argv", argv), \
             patch.dict(os.environ, {"GOOGLE_WEATHER_KEY": "FAKE"}), \
             patch("cfas.snapshot.forecast_index", return_value=fake):
            try:
                RUN.main()
            except SystemExit:
                pass

        # The run fetched a snapshot offline-first, then scored from it.
        assert (out / "snapshot.json").exists(), "an online run should leave a snapshot"
        rows = {json.loads(l)["date"]: json.loads(l)
                for l in (out / "assessments.jsonl").read_text().splitlines()}
        # Torrential forecast escalates to HIGH on rain alone, despite dry ground.
        assert rows[d0.isoformat()]["band"] == "HIGH", rows[d0.isoformat()]
        # Dry forecast stays LOW.
        assert rows[d1.isoformat()]["band"] == "LOW", rows[d1.isoformat()]
        # A broadcast sheet exists only for the day at or above MEDIUM.
        sheets = list(out.glob("*_broadcast.txt"))
        assert len(sheets) == 1, [s.name for s in sheets]
        # The broadcast does not claim wet ground when soil was dry.
        text = sheets[0].read_text().lower()
        assert "already soaked" not in text, "advisory claimed wet ground on dry soil"
    print("PASS test_forecast_drives_pipeline_end_to_end")


if __name__ == "__main__":
    test_forecast_drives_pipeline_end_to_end()
