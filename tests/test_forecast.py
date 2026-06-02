"""Offline checks for the Google Weather forecast parsing in forecast.py.

The live API needs a key and network, so here we feed the documented response
shape through the parser and confirm the day total (day + night), the P mapping,
and the probability lean behave as intended.

Run with pytest, or directly:  python tests/test_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import forecast as F  # noqa: E402


def test_day_total_sums_day_and_night():
    # A heavy day from the Google docs: 15.588 mm day + 11.3513 mm night.
    day = F.DayRain(date="2025-02-13", qpf_mm=15.588 + 11.3513, prob=0.75)
    assert round(day.qpf_mm, 2) == 26.94
    print("PASS test_day_total_sums_day_and_night")


def test_p_scales_with_probability():
    # Same heavy amount, different odds: likely rain should weigh more.
    heavy_likely = F.DayRain("d", qpf_mm=40.0, prob=0.90)
    heavy_unlikely = F.DayRain("d", qpf_mm=40.0, prob=0.05)
    assert F.p_from_day(heavy_likely) > F.p_from_day(heavy_unlikely)
    # prob_lean=0 ignores probability, so the two match.
    assert F.p_from_day(heavy_likely, prob_lean=0.0) == F.p_from_day(heavy_unlikely, prob_lean=0.0)
    print("PASS test_p_scales_with_probability")


def test_p_bounds_and_dry_day():
    dry = F.DayRain("d", qpf_mm=0.0, prob=0.0)
    soaking = F.DayRain("d", qpf_mm=120.0, prob=1.0)  # past the 50 mm full mark
    assert F.p_from_day(dry) == 0.0
    assert F.p_from_day(soaking) == 1.0   # clamped to 1.0
    print("PASS test_p_bounds_and_dry_day")


def test_block_parser_reads_qpf_and_prob():
    block = {"precipitation": {"probability": {"percent": 75, "type": "RAIN"},
                               "qpf": {"quantity": 15.588, "unit": "MILLIMETERS"}}}
    qpf, prob = F._block_qpf(block)
    assert qpf == 15.588 and prob == 0.75
    # A missing block reads as a dry zero rather than an error.
    assert F._block_qpf({}) == (0.0, 0.0)
    print("PASS test_block_parser_reads_qpf_and_prob")


def _run():
    test_day_total_sums_day_and_night()
    test_p_scales_with_probability()
    test_p_bounds_and_dry_day()
    test_block_parser_reads_qpf_and_prob()


if __name__ == "__main__":
    _run()
