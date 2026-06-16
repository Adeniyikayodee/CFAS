"""Call-in transcription backends: N-ATLAS leads when it is configured, and the
chain falls through to Cactus and Whisper when it is away. These inject fakes, so
they run with no network, no transformers download, and no audio on disk.

Run with pytest, or directly:  python tests/test_transcribe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas.advisory import Advisor  # noqa: E402


def test_natlas_leads_when_configured():
    adv = Advisor(natlas_model="ncair/n-atlas")
    # Pre-load a fake ASR pipeline so transformers is never imported or downloaded.
    adv._natlas = lambda path: {"text": "  ruwa ya cika hanya  "}
    adv._stt_cactus = lambda p, h: "cactus result"   # should be skipped
    adv._stt_local = lambda p, h: "whisper result"    # should be skipped
    assert adv.transcribe("callin.wav", "ha") == "ruwa ya cika hanya"
    print("PASS test_natlas_leads_when_configured")


def test_falls_through_to_whisper_when_natlas_absent():
    adv = Advisor()                              # no NATLAS_MODEL
    adv._stt_cactus = lambda p, h: None          # cactus away
    adv._stt_local = lambda p, h: "whisper result"
    assert adv.transcribe("callin.wav") == "whisper result"
    print("PASS test_falls_through_to_whisper_when_natlas_absent")


def test_natlas_failure_falls_through():
    # A configured model that fails to load returns None, so the chain continues.
    adv = Advisor(natlas_model="ncair/n-atlas")
    def _boom(path):
        raise RuntimeError("model unavailable on this box")
    adv._natlas = _boom
    adv._stt_cactus = lambda p, h: None
    adv._stt_local = lambda p, h: "whisper result"
    assert adv.transcribe("callin.wav", "ha") == "whisper result"
    print("PASS test_natlas_failure_falls_through")


def test_explicit_transcriber_overrides_everything():
    adv = Advisor(natlas_model="ncair/n-atlas",
                  transcriber=lambda path, hint=None: "scripted")
    assert adv.transcribe("callin.wav") == "scripted"
    print("PASS test_explicit_transcriber_overrides_everything")


def _run():
    test_natlas_leads_when_configured()
    test_falls_through_to_whisper_when_natlas_absent()
    test_natlas_failure_falls_through()
    test_explicit_transcriber_overrides_everything()


if __name__ == "__main__":
    _run()
