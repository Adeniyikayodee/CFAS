"""Adversarial checks. These do not confirm the happy path; they hunt for the
ways the system misleads, saturates, or breaks under conditions a real deployment
will meet. Each test states the disaster-domain or engineering concern it probes.

Run:  python tests/test_critical.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import forecast as F  # noqa: E402
from cfas import risk as R      # noqa: E402

W = (0.40, 0.30, 0.30)
findings = []


def note(msg):
    findings.append(msg)
    print("  FINDING:", msg)


def assess(p, v, theta):
    R.rainfall_p = lambda *a, **k: (p, p * 50.0)
    R.soil_theta = lambda *a, **k: (theta, theta * 0.5)
    R.vulnerability_v = lambda *a, **k: v
    return R.assess(None, None, 0.0, 0.0, "2025-01-01", "2025-01-02",
                    weights=W, tessera_year=2024, scale=1000)


# 1. DOMAIN: a dry forecast on already-saturated ground over a flood plain.
#    Antecedent saturation is the classic flash-flood setup; a system that only
#    fires on forecast rain will miss it. What does CFAS do?
def test_saturated_ground_no_forecast_rain():
    out = assess(p=0.0, v=1.0, theta=1.0)
    # raw = .4*0 + .3*1 + .3*1*mu(1)=.3*2=.6 -> .9 ; /1.3 = .69 -> HIGH
    print(f"  saturated+exposed, zero rain -> {out.band} (score {out.score:.2f})")
    assert out.band in ("MEDIUM", "HIGH"), "saturated flood-plain ground should not read LOW"
    if out.band != "HIGH":
        note("Saturated, highly exposed ground with no forecast rain reads only "
             f"{out.band}. Antecedent-saturation flash floods may underscore.")


# 2. DOMAIN/STATS: can rainfall alone ever raise a HIGH warning?
#    With a=0.40, even a catastrophic P=1.0 contributes 0.40/1.30 = 0.31 to score.
#    A pure rainfall extreme on safe, dry ground cannot cross 0.34. That is a
#    serious miss for a flood warning system.
def test_extreme_rain_alone():
    out = assess(p=1.0, v=0.0, theta=0.0)
    print(f"  extreme rain, safe dry ground -> {out.band} (score {out.score:.2f})")
    if R.RANK[out.band] < R.RANK["MEDIUM"]:
        note("A maximal rainfall forecast on low-vulnerability dry ground cannot "
             f"reach MEDIUM (score {out.score:.2f}). With a=0.40 the rain term caps "
             "at 0.31 of the normalised score, so rain alone never warns. For a "
             "FLOOD system this is the most important failure to weigh.")


# 3. STATS: the normaliser divides by (a+b+2g). Is the score reachable to 1.0,
#    and is it monotone in each input? A score that cannot span its band range
#    wastes resolution.
def test_score_reachability_and_monotonicity():
    hi = assess(p=1.0, v=1.0, theta=1.0).score
    lo = assess(p=0.0, v=0.0, theta=0.0).score
    print(f"  score range observed: [{lo:.2f}, {hi:.2f}]")
    assert lo == 0.0
    assert abs(hi - 1.0) < 1e-9, "max inputs should reach score 1.0"
    # monotone in rain
    a1 = assess(0.2, 0.5, 0.5).score
    a2 = assess(0.8, 0.5, 0.5).score
    assert a2 > a1, "score must rise with rainfall"


# 4. DOMAIN: the mu multiplier is flat below theta0=0.6, then ramps. A field that
#    sits at 0.59 vs 0.61 should not flip risk wildly. Check for a cliff.
def test_mu_continuity():
    below = assess(0.3, 0.3, 0.59).score
    above = assess(0.3, 0.3, 0.61).score
    print(f"  theta 0.59 vs 0.61 -> {below:.3f} vs {above:.3f}")
    assert abs(above - below) < 0.05, "mu should be continuous, not a cliff at theta0"


# 5. ENGINEERING: forecast P with a NaN / null qpf from the API must not poison
#    the score. Real APIs return nulls.
def test_forecast_handles_null_fields():
    block = {"precipitation": {"probability": {"percent": None},
                               "qpf": {"quantity": None}}}
    qpf, prob = F._block_qpf(block)
    assert qpf == 0.0 and prob == 0.0, "null qpf/prob must read as 0, not crash"
    day = F.DayRain("d", qpf_mm=qpf, prob=prob)
    assert F.p_from_day(day) == 0.0


# 6. ENGINEERING: a negative or absurd qpf (bad upstream data) must clamp, not
#    produce a negative P that silently lowers a real warning.
def test_forecast_rejects_absurd_values():
    p_neg = F.p_from_day(F.DayRain("d", qpf_mm=-10.0, prob=0.5))
    p_huge = F.p_from_day(F.DayRain("d", qpf_mm=99999.0, prob=1.0))
    print(f"  P(neg qpf)={p_neg}, P(huge qpf)={p_huge}")
    assert 0.0 <= p_neg <= 1.0 and 0.0 <= p_huge <= 1.0
    if p_neg < 0.0:
        note("Negative qpf produces negative P, which would suppress risk.")


# 7. ENGINEERING: forecast_index keys by date. If the API returns a malformed
#    displayDate (missing fields), we must not emit a key like '0000-00-00' that
#    silently fails to match the run loop's dates.
def test_forecast_malformed_date_is_visible():
    sample = {"forecastDays": [
        {"daytimeForecast": {"precipitation": {"qpf": {"quantity": 30.0},
                                               "probability": {"percent": 90}}},
         "nighttimeForecast": {}}]}  # no displayDate

    class Resp:
        def raise_for_status(self): pass
        def json(self): return sample
    raised = False
    with patch("requests.get", return_value=Resp()):
        try:
            F.forecast_index(7.8, 6.7, key="X", days=1)
        except ValueError:
            raised = True
    print(f"  malformed date raised ValueError: {raised}")
    assert raised, "a malformed forecast date must fail loudly, not emit a phantom key"


# 8. DISASTER DOMAIN: band thresholds vs action. MEDIUM triggers a broadcast that
#    tells people to move grain and animals; HIGH tells people to move themselves.
#    A system biased toward false alarms erodes trust (the cry-wolf effect). Check
#    where a "typical wet-season day" lands: moderate rain, moderate vulnerability,
#    seasonally wet soil. It should NOT be HIGH every day, or trust collapses.
def test_typical_wet_season_day_is_not_high():
    out = assess(p=0.5, v=0.5, theta=0.55)
    print(f"  typical wet-season day -> {out.band} (score {out.score:.2f})")
    if out.band == "HIGH":
        note("A merely typical wet-season day reads HIGH. Daily HIGH warnings "
             "trigger evacuation-level messaging and invite cry-wolf fatigue.")


def test_rain_trigger_band_score_consistent():
    # After the rainfall floor, band and score must never contradict each other,
    # or an operator sees "HIGH, score 0.2" and stops trusting the system.
    for p in (0.60, 0.90, 1.0):
        out = assess(p=p, v=0.0, theta=0.0)
        if out.band == "HIGH":
            assert out.score >= 0.67, f"HIGH paired with score {out.score}"
        elif out.band == "MEDIUM":
            assert 0.34 <= out.score < 0.67, f"MEDIUM paired with score {out.score}"
    print("PASS test_rain_trigger_band_score_consistent")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
            print("  ok")
            passed += 1
        except AssertionError as e:
            print("  HARD FAIL:", e)
    print(f"\n{'='*64}\n{passed}/{len(tests)} structural checks passed")
    print(f"{len(findings)} domain/engineering finding(s) flagged for judgement")
    if findings:
        print("\nFINDINGS TO WEIGH:")
        for i, f in enumerate(findings, 1):
            print(f"  {i}. {f}")


if __name__ == "__main__":
    _run()
