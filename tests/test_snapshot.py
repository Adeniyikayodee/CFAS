"""Offline-first snapshot: the cached replay must band a day exactly as the live
read would, the freshness guard must fire, and an --offline run must never reach
the network. These run with the standard library alone, no keys, no downloads.

The parity checks are the ones that matter most: live and offline scoring share
risk.fuse(), so a band computed from the snapshot equals the band computed live
from the same inputs. Calibration joins the two ledgers, so any drift between
them would quietly corrupt the hit-rate.

Run with pytest, or directly:  python tests/test_snapshot.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import risk as R          # noqa: E402
from cfas import run as RUN         # noqa: E402
from cfas import snapshot as SNAP   # noqa: E402
from cfas.forecast import DayRain, p_from_day  # noqa: E402

W = (0.40, 0.30, 0.30)


def _mk_snap(days=None, fetched_at=None, embedding=None):
    return {
        "schema": SNAP.SCHEMA,
        "fetched_at": fetched_at or dt.datetime.now().isoformat(timespec="seconds"),
        "location": {"community": "Adankolo", "subcounty": "Lokoja",
                     "county": "Kogi", "country": "Nigeria", "lat": 7.8, "lon": 6.74},
        "tessera": {"year": 2024, "embedding": embedding},
        "days": days if days is not None else {
            "2026-06-10": {"qpf_mm": 40.0, "prob": 1.0, "soil": 0.35, "source": "gfs/chirps"}},
    }


def test_assess_delegates_to_fuse():
    # The live path must add no banding logic of its own beyond gathering inputs.
    R.rainfall_p = lambda *a, **k: (0.8, 40.0)
    R.soil_theta = lambda *a, **k: (0.7, 0.35)
    R.fetch_embedding = lambda *a, **k: None          # V falls back to the proxy default
    live = R.assess(None, None, 0.0, 0.0, "2026-06-10", "2026-06-11",
                    weights=W, tessera_year=2024, scale=1000)
    assert live == R.fuse(0.8, 0.5, 0.7, weights=W, rain_mm=40.0, soil=0.35)
    print("PASS test_assess_delegates_to_fuse")


def test_live_and_snapshot_agree_on_the_same_inputs():
    # The headline guarantee: a GFS day read live and the same day replayed from
    # the snapshot produce an identical Risk. With prob = 1.0 the snapshot's
    # prob-weighted P collapses to the amount-only P the live GFS path computes.
    R.rainfall_p = lambda *a, **k: (0.8, 40.0)
    R.soil_theta = lambda *a, **k: (0.7, 0.35)
    R.fetch_embedding = lambda *a, **k: None
    live = R.assess(None, None, 0.0, 0.0, "2026-06-10", "2026-06-11",
                    weights=W, tessera_year=2024, scale=1000)
    offline = SNAP.risk_for_day(_mk_snap(), "2026-06-10", weights=W, prob_lean=0.35)
    assert live == offline, (live, offline)
    print("PASS test_live_and_snapshot_agree_on_the_same_inputs")


def test_snapshot_replay_recomputes_terms_through_fuse():
    snap = _mk_snap(days={"2026-06-10": {"qpf_mm": 22.0, "prob": 0.8,
                                         "soil": 0.30, "source": "google-weather"}})
    got = SNAP.risk_for_day(snap, "2026-06-10", weights=W, prob_lean=0.35)
    p = p_from_day(DayRain("2026-06-10", 22.0, 0.8), 0.35)
    theta = min(0.30 / R.SOIL_FULL, 1.0)
    expect = R.fuse(p, R.v_from_embedding(None), theta, weights=W, rain_mm=22.0, soil=0.30)
    assert got == expect, (got, expect)
    print("PASS test_snapshot_replay_recomputes_terms_through_fuse")


def test_retuning_weights_rebands_offline_without_refetch():
    # The point of caching raw inputs: turning the dials re-bands the same snapshot.
    snap = _mk_snap(days={"2026-06-10": {"qpf_mm": 18.0, "prob": 0.7,
                                         "soil": 0.45, "source": "google-weather"}})
    soil_heavy = SNAP.risk_for_day(snap, "2026-06-10", weights=(0.1, 0.1, 0.8), prob_lean=0.35)
    rain_heavy = SNAP.risk_for_day(snap, "2026-06-10", weights=(0.8, 0.1, 0.1), prob_lean=0.35)
    assert soil_heavy.score != rain_heavy.score, "weights should move the score offline"
    print("PASS test_retuning_weights_rebands_offline_without_refetch")


def test_missing_day_returns_none():
    assert SNAP.risk_for_day(_mk_snap(), "1999-01-01", weights=W) is None
    print("PASS test_missing_day_returns_none")


def test_save_load_roundtrip_and_schema_guard(tmp_path: Path):
    snap = _mk_snap(embedding=[0.1, -0.2, 0.3])
    path = tmp_path / "snapshot.json"
    SNAP.save(path, snap)
    assert SNAP.load(path) == snap
    assert SNAP.load(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "cfas-snapshot/99", "days": {}}))
    raised = False
    try:
        SNAP.load(bad)
    except ValueError:
        raised = True
    assert raised, "an unknown schema must fail loudly, not load silently"
    print("PASS test_save_load_roundtrip_and_schema_guard")


def test_staleness_guard():
    now = dt.datetime(2026, 6, 16, 12, 0, 0)
    snap = _mk_snap(fetched_at=(now - dt.timedelta(hours=30)).isoformat(timespec="seconds"))
    assert abs(SNAP.age_hours(snap, now) - 30.0) < 0.01
    assert SNAP.is_stale(snap, 24, now) is True
    assert SNAP.is_stale(snap, 48, now) is False
    print("PASS test_staleness_guard")


class _FakeAdvisor:
    # Stands in for the real Advisor so the offline run needs no NLLB, no gTTS and
    # no network; the snapshot path, not drafting, is what this test exercises.
    def __init__(self, **kwargs):
        pass

    def draft(self, ctx):
        return f"Test advisory for {ctx['community']}, band {ctx['band']}."

    def translate(self, text, tgt, src=None):
        return text

    def voice(self, text, gtts_code, path):
        return None


def test_offline_run_scores_from_snapshot_without_network(tmp_path: Path):
    # init_ee raising proves --offline never reaches Earth Engine: a fresh snapshot
    # is used directly, so the build path (and its network reads) is never entered.
    def _boom(*a, **k):
        raise AssertionError("offline run must not touch Earth Engine")
    R.init_ee = _boom

    out = tmp_path / "alerts"
    out.mkdir()
    SNAP.save(out / "snapshot.json", _mk_snap(days={
        "2026-06-10": {"qpf_mm": 60.0, "prob": 0.95, "soil": 0.05, "source": "google-weather"}}))

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "location: {country: Nigeria, county: Kogi, subcounty: Lokoja, "
        "community: Adankolo, landmark: the market bridge}\n"
        "window: {start: 2026-06-10, end: 2026-06-10, expected_hour: this evening}\n"
        "weights: {alpha: 0.40, beta: 0.30, gamma: 0.30}\n"
        "languages: [hausa]\nalert_from: MEDIUM\n")

    argv = ["run", "--config", str(cfg), "--outdir", str(out), "--offline"]
    with patch.object(sys, "argv", argv), patch("cfas.run.Advisor", _FakeAdvisor):
        try:
            RUN.main()
        except SystemExit:
            pass

    rows = [json.loads(l) for l in (out / "assessments.jsonl").read_text().splitlines()]
    assert rows and rows[-1]["band"] == "HIGH", rows
    assert list(out.glob("*_broadcast.txt")), "offline run should still write a broadcast sheet"
    print("PASS test_offline_run_scores_from_snapshot_without_network")


def _run():
    test_assess_delegates_to_fuse()
    test_live_and_snapshot_agree_on_the_same_inputs()
    test_snapshot_replay_recomputes_terms_through_fuse()
    test_retuning_weights_rebands_offline_without_refetch()
    test_missing_day_returns_none()
    test_staleness_guard()
    for test in (test_save_load_roundtrip_and_schema_guard,
                 test_offline_run_scores_from_snapshot_without_network):
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))


if __name__ == "__main__":
    _run()
