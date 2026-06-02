"""Measure the warnings against what the community confirmed.

The warning loop writes one band per day to alerts/assessments.jsonl. The call-in
loop writes listener reports to alerts/feedback.jsonl, where a native speaker marks
each one flood or dry. This joins the two by community and a nearby date, then
reports the four outcomes and the rates that matter for a warning system:

    hit-rate (recall), precision, and false-alarm rate.

The contingency-table method, with hits, misses, false alarms and correct calms,
is the standard way to score a categorical forecast (Jolliffe & Stephenson,
"Forecast Verification: A Practitioner's Guide in Atmospheric Science," 2nd ed.,
Wiley, 2012). The headline target of 65 to 75 percent ties back to the call for
early-warning systems that reach the people most exposed (WMO & UNDRR, "Global
Status of Multi-Hazard Early Warning Systems," 2023).

    python -m cfas.calibrate                       # reads ./alerts
    python -m cfas.calibrate --alert-band HIGH      # stricter view
    python -m cfas.calibrate --window-days 2        # wider date match
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .risk import RANK

ROOT = Path(__file__).resolve().parent.parent
TRUE = {"true", "yes", "y", "1", "flood", "flooded", "confirmed"}
FALSE = {"false", "no", "n", "0", "dry", "clear", "calm"}


def parse_confirmed(value):
    """Read a reviewer's mark as flood (True), dry (False), or pending (None)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    return True if token in TRUE else False if token in FALSE else None


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_bands(rows):
    """Keep one band per community and date, last write wins."""
    keep = {}
    for r in rows:
        keep[(r["community"], r["date"])] = r
    return list(keep.values())


def nearest_band(report, bands, window):
    target = dt.date.fromisoformat(report["date"])
    best, best_gap = None, window + 1
    for b in bands:
        if b["community"] != report["community"]:
            continue
        gap = abs((dt.date.fromisoformat(b["date"]) - target).days)
        if gap <= window and gap < best_gap:
            best, best_gap = b, gap
    return best


def calibrate(alerts_dir: Path, alert_band: str, window: int):
    bands = latest_bands(load_jsonl(alerts_dir / "assessments.jsonl"))
    feedback = load_jsonl(alerts_dir / "feedback.jsonl")
    threshold = RANK[alert_band]
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "reviewed": 0, "matched": 0, "unmatched": 0}
    for report in feedback:
        truth = parse_confirmed(report.get("confirmed"))
        if truth is None:
            continue
        c["reviewed"] += 1
        band = nearest_band(report, bands, window)
        if band is None:
            c["unmatched"] += 1
            continue
        c["matched"] += 1
        alerted = RANK.get(band["band"], 0) >= threshold
        key = ("tp" if truth else "fp") if alerted else ("fn" if truth else "tn")
        c[key] += 1
    return c


def ratio(a, b):
    return a / b if b else None


def metrics(c):
    # Standard scores from the contingency table (Jolliffe & Stephenson, 2012):
    # recall is the share of real floods we warned about, precision the share of
    # warnings that proved real, and the false-alarm rate the share of dry days
    # that drew a warning. Each guards against an empty denominator.
    return {
        "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "false_alarm": ratio(c["fp"], c["fp"] + c["tn"]),
        "specificity": ratio(c["tn"], c["tn"] + c["fp"]),
    }


def pct(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def main():
    ap = argparse.ArgumentParser(description="Calibrate CFAS warnings against confirmed call-ins")
    ap.add_argument("--alerts", default=str(ROOT / "alerts"))
    ap.add_argument("--alert-band", default="MEDIUM", choices=["MEDIUM", "HIGH"])
    ap.add_argument("--window-days", type=int, default=1)
    args = ap.parse_args()

    d = Path(args.alerts)
    d.mkdir(parents=True, exist_ok=True)
    c = calibrate(d, args.alert_band, args.window_days)
    m = metrics(c)

    print(f"""
CFAS calibration   alert at {args.alert_band}+, date match within {args.window_days} day(s)
============================================================
confirmed call-ins reviewed : {c['reviewed']}
matched to an assessment    : {c['matched']}        pending a match: {c['unmatched']}

                  flood            dry
   warned     {c['tp']:>4} hit       {c['fp']:>4} false alarm
   quiet      {c['fn']:>4} miss      {c['tn']:>4} correct calm

   hit-rate (recall) : {pct(m['recall'])}
   precision         : {pct(m['precision'])}
   false-alarm rate  : {pct(m['false_alarm'])}
   specificity       : {pct(m['specificity'])}
""")
    if c["reviewed"] == 0:
        print("Mark the `confirmed` field in alerts/feedback.jsonl as flood or dry to begin.\n")

    (d / "calibration.json").write_text(json.dumps(
        {"counts": c, "metrics": m, "alert_band": args.alert_band,
         "window_days": args.window_days}, indent=2))


if __name__ == "__main__":
    main()
