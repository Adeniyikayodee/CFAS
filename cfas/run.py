"""CFAS entrypoint: config and keys in, broadcast-ready warnings out.

This module is the conductor. It reads the satellite layers through risk.py, hands
the score to advisory.py for drafting and translation, writes the broadcast files,
and keeps the per-day band ledger that calibrate.py later scores. The pipeline and
its three-band, radio-first design are the CFAS work (Adeniyi, CFAS, 2026), built
on the finding that most exposed communities still wait for warnings they can act
on (WMO & UNDRR, "Global Status of Multi-Hazard Early Warning Systems," 2023).

    python -m cfas.run                          # uses config.yaml + .env
    python -m cfas.run --config x.yaml
    python -m cfas.run --dry-run                # delivery half only, skips Earth Engine
    python -m cfas.run --listen ./callins --feedback-lang hausa   # the call-in loop
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import risk as R
from .advisory import ACTIONS, LANGS, SOURCE, Advisor
from .forecast import forecast_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cfas")
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    country: str; county: str; subcounty: str; community: str; landmark: str
    start: dt.date; end: dt.date; hour: str
    weights: tuple; tessera_year: int; buffer_km: float; scale_m: int
    alert_from: str; languages: list; prob_lean: float
    trigger_medium: float; trigger_high: float


def load_settings(path: Path) -> Settings:
    cfg = yaml.safe_load(path.read_text())
    loc, win, w = cfg["location"], cfg["window"], cfg["weights"]
    langs = [l for l in cfg.get("languages", ["hausa"]) if l in LANGS] or ["hausa"]
    return Settings(
        country=loc["country"], county=loc["county"], subcounty=loc["subcounty"],
        community=loc["community"], landmark=loc.get("landmark", "the nearest high ground"),
        start=dt.date.fromisoformat(str(win["start"])), end=dt.date.fromisoformat(str(win["end"])),
        hour=win.get("expected_hour", "this evening"),
        weights=(w["alpha"], w["beta"], w["gamma"]), tessera_year=int(cfg.get("tessera_year", 2024)),
        buffer_km=float(cfg.get("aoi_buffer_km", 20)), scale_m=int(cfg.get("scale_m", 1000)),
        alert_from=cfg.get("alert_from", "MEDIUM").upper(), languages=langs,
        prob_lean=float(cfg.get("forecast_prob_lean", 0.35)),
        trigger_medium=float(cfg.get("rain_trigger_medium", 0.60)),
        trigger_high=float(cfg.get("rain_trigger_high", 0.90)))


def load_keys():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def emit(s: Settings, day: str, risk: R.Risk, advisor: Advisor, outdir: Path):
    ctx = {"community": s.community, "subcounty": s.subcounty, "county": s.county,
           "country": s.country, "landmark": s.landmark, "band": risk.band, "date": day,
           "hour": s.hour, "rain_mm": risk.rain_mm, "soil": risk.soil, "action": ACTIONS[risk.band]}
    english = advisor.draft(ctx)
    versions = {code: advisor.translate(english, LANGS[code][1]) for code in s.languages}

    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{s.country}_{s.community}_{day.replace('-', '')}".replace(" ", "")
    record = {"place": ctx, "risk": risk.__dict__, "english": english, "versions": versions,
              "model": "R = a*P + b*V(TESSERA) + g*theta*mu(theta)"}
    (outdir / f"{base}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))

    sheet = [f"CFAS FLOOD ADVISORY  {s.community}, {s.subcounty}, {s.county}",
             f"{day}   band {risk.band}   score {risk.score:.2f}",
             f"rain {risk.rain_mm} mm | soil {risk.soil} | vulnerability {risk.vuln}",
             "=" * 64, "", "[English]", english, ""]
    for code in s.languages:
        sheet += [f"[{LANGS[code][0]}]", versions[code], ""]
        advisor.voice(versions[code], LANGS[code][2], str(outdir / f"{base}_{code}.mp3"))
    (outdir / f"{base}_broadcast.txt").write_text("\n".join(sheet))

    print("\n" + "=" * 64)
    print(f"{s.community} ({s.subcounty})  {day}  [{risk.band}]")
    print("English:", english)
    for code in s.languages:
        print(f"{LANGS[code][0]}:", versions[code])
    print("=" * 64)


AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


def listen(advisor: Advisor, audio_dir: Path, lang_code: str | None, outdir: Path,
           community: str, obs_date: str):
    """Transcribe listener call-ins and carry them back to English for the team.

    Drop voice notes into a folder, point `--feedback-lang` at the station's
    language, and each line lands in alerts/feedback.jsonl for review. The
    `confirmed` field waits for a native speaker to mark flood or dry, which is
    the ground truth the calibrator reads. Folding the community's own reports back
    into the system follows the case for citizen science in hydrology (Buytaert
    et al., Front. Earth Sci. 2:26, 2014).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "feedback.jsonl"
    lang = LANGS.get(lang_code) if lang_code else None
    rows = 0
    for path in sorted(p for p in Path(audio_dir).glob("*") if p.suffix.lower() in AUDIO_EXT):
        transcript = advisor.transcribe(str(path), lang_hint=lang.iso if lang else None)
        english = advisor.translate(transcript, SOURCE, src=lang.flores) if (lang and transcript) else transcript
        record = {"file": path.name, "community": community, "date": obs_date,
                  "language": lang_code or "auto", "transcript": transcript, "english": english,
                  "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                  "confirmed": None}  # the native-speaker reviewer marks flood or dry
        with ledger.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("%s  ->  %s", path.name, english or transcript or "(silence)")
        rows += 1
    print(f"\nLogged {rows} call-in(s) to {ledger}")


def main():
    ap = argparse.ArgumentParser(description="CFAS flood-resilience pipeline")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--outdir", default=str(ROOT / "alerts"))
    ap.add_argument("--vuln-head", help="path to a trained (129,) linear probe for V")
    ap.add_argument("--dry-run", action="store_true", help="delivery half only, skips Earth Engine")
    ap.add_argument("--listen", metavar="DIR", help="transcribe listener call-ins from a folder and log them")
    ap.add_argument("--feedback-lang", choices=list(LANGS), help="language of the call-ins (sets STT hint and back-translation)")
    ap.add_argument("--feedback-date", help="observation date for the call-ins (default today)")
    args = ap.parse_args()

    load_keys()
    s = load_settings(Path(args.config))
    advisor = Advisor(cactus_url=os.environ.get("CACTUS_URL"),
                      gemma_model=os.environ.get("GEMMA_MODEL", "gemma-3-it.gguf"),
                      nllb_model=os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
                      stt_model=os.environ.get("STT_MODEL", "base"))

    if args.listen:
        obs = args.feedback_date or dt.date.today().isoformat()
        listen(advisor, Path(args.listen), args.feedback_lang, Path(args.outdir), s.community, obs)
        return

    head = np.load(args.vuln_head) if args.vuln_head else None
    log.info("%s, %s | %s to %s | languages: %s", s.community, s.country, s.start, s.end,
             ", ".join(LANGS[l][0] for l in s.languages))

    ee = aoi = lat = lon = None
    forecast = {}
    if not args.dry_run:
        ee = R.init_ee(os.environ.get("EE_PROJECT"))
        lat, lon = R.geocode(s.country, s.county, s.subcounty, s.community)
        aoi = R.aoi_of(ee, lat, lon, s.buffer_km)
        log.info("AOI %.4f, %.4f  (%g km)", lat, lon, s.buffer_km)

        # P looks ahead with the Google Weather forecast when a key is present.
        # Without it, assess reads the Earth Engine GFS forecast instead.
        gkey = os.environ.get("GOOGLE_WEATHER_KEY")
        if gkey:
            try:
                span = (s.end - s.start).days + 1
                forecast = forecast_index(lat, lon, gkey, days=span, prob_lean=s.prob_lean)
                log.info("Google Weather forecast loaded for %d day(s)", len(forecast))
            except Exception as e:
                log.warning("Google Weather forecast unavailable (%s); using GFS", e)
        else:
            log.info("No GOOGLE_WEATHER_KEY set; P will read Earth Engine GFS")

    outdir, issued, day = Path(args.outdir), 0, s.start
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "assessments.jsonl"
    while day <= s.end:
        d = day.isoformat()
        if args.dry_run:
            risk = R.Risk(0.72, "HIGH", 62.0, 0.41, 0.66, 0.9, 0.82)
        else:
            risk = R.assess(ee, aoi, lat, lon, d, (day + dt.timedelta(days=1)).isoformat(),
                            weights=s.weights, tessera_year=s.tessera_year, scale=s.scale_m,
                            head=head, forecast_p=forecast.get(d),
                            trigger_medium=s.trigger_medium, trigger_high=s.trigger_high)
        log.info("%s  band=%s score=%.2f  (rain=%s theta=%s V=%s)",
                 d, risk.band, risk.score, risk.rain_mm, risk.theta, risk.vuln)
        with ledger.open("a") as fh:
            fh.write(json.dumps({"community": s.community, "country": s.country,
                                 "subcounty": s.subcounty, "date": d,
                                 "band": risk.band, "score": round(risk.score, 3)}) + "\n")
        if R.RANK[risk.band] >= R.RANK[s.alert_from]:
            emit(s, d, risk, advisor, outdir)
            issued += 1
        day += dt.timedelta(days=1)

    print(f"\nIssued {issued} advisory day(s) at or above {s.alert_from}. Files in {outdir}")


if __name__ == "__main__":
    main()
