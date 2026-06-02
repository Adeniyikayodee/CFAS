"""Offline checks for the flood-risk processing in risk.py.

These exercise the math that turns the three satellite layers into a banded
score: the saturation multiplier, the band cutoffs, and the full fusion in
`assess`. The live satellite reads (GFS, SMAP, TESSERA) sit behind Earth Engine
and GeoTessera, so here we feed `assess` known layer values and confirm the
arithmetic and bands land where they should.

Run with pytest, or directly:  python tests/test_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import risk as R  # noqa: E402

W = (0.40, 0.30, 0.30)


def test_saturation_multiplier():
    # Below the saturation point the multiplier rests at 1.0; it climbs to 2.0 as
    # the ground fills, which is how soaked soil lifts the soil-moisture term.
    assert R.mu(0.30) == 1.0
    assert R.mu(0.60) == 1.0
    assert round(R.mu(0.80), 2) == 1.50
    assert R.mu(1.00) == 2.0
    assert R.mu(0.50) <= R.mu(0.90)  # monotone rising
    print("PASS test_saturation_multiplier")


def test_bands():
    assert R.band_of(0.10) == "LOW"
    assert R.band_of(0.33) == "LOW"
    assert R.band_of(0.34) == "MEDIUM"
    assert R.band_of(0.50) == "MEDIUM"
    assert R.band_of(0.67) == "HIGH"
    assert R.band_of(0.95) == "HIGH"
    print("PASS test_bands")


def _assess_with(p, v, theta, monkeypatch_targets):
    # Stand in for the three live layers so we test the fusion alone.
    R.rainfall_p = lambda *a, **k: (p, p * 50.0)
    R.soil_theta = lambda *a, **k: (theta, theta * 0.5)
    R.vulnerability_v = lambda *a, **k: v
    return R.assess(None, None, 0.0, 0.0, "2024-09-01", "2024-09-02",
                    weights=W, tessera_year=2024, scale=1000)


def test_fusion_high():
    # Strong rain, exposed ground, soaked soil -> HIGH (matches the dry-run figure).
    out = _assess_with(0.90, 0.66, 0.82, None)
    # raw = .4*.9 + .3*.66 + .3*.82*mu(.82);  score = raw / (a+b+2g)
    expected = (0.4 * 0.9 + 0.3 * 0.66 + 0.3 * 0.82 * R.mu(0.82)) / (0.4 + 0.3 + 0.6)
    assert abs(out.score - expected) < 1e-9
    assert out.band == "HIGH"
    print(f"PASS test_fusion_high (score={out.score:.3f})")


def test_fusion_low():
    # Dry forecast, low ground, dry soil -> LOW, and no warning would go out.
    out = _assess_with(0.05, 0.10, 0.10, None)
    assert out.band == "LOW"
    assert out.score < 0.34
    print(f"PASS test_fusion_low (score={out.score:.3f})")


def test_fusion_medium():
    # A middling day lands in the MEDIUM band, where a broadcast begins.
    out = _assess_with(0.60, 0.50, 0.60, None)
    assert out.band == "MEDIUM"
    assert 0.34 <= out.score < 0.67
    print(f"PASS test_fusion_medium (score={out.score:.3f})")


def _run():
    test_saturation_multiplier()
    test_bands()
    test_fusion_high()
    test_fusion_medium()
    test_fusion_low()


if __name__ == "__main__":
    _run()
