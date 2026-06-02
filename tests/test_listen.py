"""Self-contained checks for the call-in loop and the calibrator.

Two layers, both offline:

  * test_listen_real runs genuine speech-to-text on a committed audio clip with
    pocketsphinx (CMU Sphinx), so the ledger fills with real transcribed words.
    It skips cleanly when pocketsphinx is absent.
  * test_pipeline_deterministic injects a scripted transcriber, so the full path
    from call-in to ledger to calibrated metrics runs fast and identically every
    time, with the standard library alone.

Run with pytest, or directly:  python tests/test_listen.py
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas import calibrate                       # noqa: E402
from cfas.advisory import Advisor                # noqa: E402
from cfas.run import listen                      # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "callin_sample.wav"  # "go forward ten meters"


# --- a real, offline speech-to-text built on pocketsphinx -------------------
def pocketsphinx_transcriber(audio_path, lang_hint=None):
    """Decode 16 kHz mono WAV with the bundled CMU Sphinx English model."""
    from pocketsphinx import Decoder
    with wave.open(audio_path, "rb") as w:
        frames = w.readframes(w.getnframes())
    d = Decoder()
    d.start_utt()
    d.process_raw(frames, full_utt=True)
    d.end_utt()
    return d.hyp().hypstr if d.hyp() else ""


def _seed_assessment(alerts: Path, community: str, date: str, band: str):
    alerts.mkdir(parents=True, exist_ok=True)
    row = {"community": community, "date": date, "band": band, "score": 0.72}
    (alerts / "assessments.jsonl").write_text(json.dumps(row) + "\n")


def test_listen_real(tmp_path: Path):
    try:
        import pocketsphinx  # noqa: F401
    except Exception:
        print("SKIP test_listen_real: install pocketsphinx to run the real ASR check")
        return
    callins = tmp_path / "callins"
    callins.mkdir()
    (callins / "hausa_caller01.wav").write_bytes(FIXTURE.read_bytes())
    alerts = tmp_path / "alerts"

    advisor = Advisor(transcriber=pocketsphinx_transcriber)
    listen(advisor, callins, "hausa", alerts, community="Adankolo", obs_date="2024-09-02")

    rows = [json.loads(l) for l in (alerts / "feedback.jsonl").read_text().splitlines()]
    transcript = rows[0]["transcript"].lower()
    print("real transcript:", repr(transcript))
    assert transcript, "the transcript should hold real words rather than silence"
    assert any(word in transcript for word in ("forward", "meters", "ten", "go")), transcript
    print("PASS test_listen_real")


def test_pipeline_deterministic(tmp_path: Path):
    callins = tmp_path / "callins"
    callins.mkdir()
    for name in ("hausa_flood.wav", "hausa_dry.wav"):
        (callins / name).write_bytes(b"placeholder")
    alerts = tmp_path / "alerts"
    _seed_assessment(alerts, "Adankolo", "2024-09-02", "HIGH")

    scripted = {"hausa_flood.wav": "the market road is under water",
                "hausa_dry.wav": "all calm here today"}
    advisor = Advisor(transcriber=lambda path, hint=None: scripted[Path(path).name])
    listen(advisor, callins, "hausa", alerts, community="Adankolo", obs_date="2024-09-02")

    rows = {json.loads(l)["file"]: json.loads(l)
            for l in (alerts / "feedback.jsonl").read_text().splitlines()}
    assert "under water" in rows["hausa_flood.wav"]["transcript"]
    assert rows["hausa_dry.wav"]["transcript"] == "all calm here today"

    # mark ground truth and score: one HIGH-day flood (hit), one HIGH-day dry (false alarm)
    marked = []
    for name, truth in (("hausa_flood.wav", True), ("hausa_dry.wav", False)):
        r = rows[name]
        r["confirmed"] = truth
        marked.append(r)
    (alerts / "feedback.jsonl").write_text("\n".join(json.dumps(r) for r in marked) + "\n")

    counts = calibrate.calibrate(alerts, "MEDIUM", 1)
    m = calibrate.metrics(counts)
    print("counts:", counts)
    print("metrics:", m)
    assert counts["tp"] == 1 and counts["fp"] == 1 and counts["fn"] == 0 and counts["tn"] == 0
    assert m["recall"] == 1.0 and m["precision"] == 0.5
    print("PASS test_pipeline_deterministic")


def _run():
    import tempfile
    for test in (test_pipeline_deterministic, test_listen_real):
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))


if __name__ == "__main__":
    _run()
