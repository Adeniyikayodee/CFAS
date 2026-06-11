"""Offline voice fallback: when gTTS has no network, Piper must still put a
spoken warning in the broadcast folder. These checks inject a fake Piper voice,
so they run with no network, no piper install, and no model download, the way the
rest of the suite stays offline.

Run with pytest, or directly:  python tests/test_voice.py
"""
from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfas.advisory import Advisor  # noqa: E402


class _FakeVoice:
    """Stands in for a loaded Piper voice. Writes a short, valid 16 kHz WAV, which
    is enough to prove the fallback wiring without pulling in piper or a model."""

    def synthesize(self, text, wav):
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)  # one second of silence


def test_piper_fallback_writes_wav_when_gtts_offline():
    adv = Advisor(piper_voices="/voices")            # truthy, never read on disk
    adv._voice_gtts = lambda text, code, path: None  # simulate no network
    adv._piper = {"ha": _FakeVoice()}                # pre-load, so piper is never imported

    with tempfile.TemporaryDirectory() as d:
        asked = str(Path(d) / "advisory_ha.mp3")
        out = adv.voice("Sannu, ku saurara.", "ha", asked)
        # Piper renders WAV, so the name is corrected from .mp3 to .wav, not faked.
        assert out is not None, "offline fallback should still produce audio"
        assert out.endswith(".wav"), out
        assert Path(out).exists()
        with wave.open(out) as w:
            assert w.getnframes() > 0, "the WAV should hold real audio frames"
    print("PASS test_piper_fallback_writes_wav_when_gtts_offline")


def test_voice_returns_none_when_no_engine():
    # No network for gTTS and no Piper voices configured: voice reports failure
    # honestly so run.py can log that only the text sheet was written.
    adv = Advisor(piper_voices=None)
    adv._voice_gtts = lambda text, code, path: None
    out = adv.voice("Sannu", "ha", "/tmp/cfas_no_engine.mp3")
    assert out is None, out
    print("PASS test_voice_returns_none_when_no_engine")


def test_gtts_success_keeps_mp3_path():
    # When the online voice succeeds, the path comes back unchanged as .mp3 and
    # Piper is never consulted (left unset, so reaching it would return None).
    adv = Advisor(piper_voices=None)
    adv._voice_gtts = lambda text, code, path: path
    out = adv.voice("Sannu", "ha", "/tmp/cfas_ok.mp3")
    assert out == "/tmp/cfas_ok.mp3", out
    print("PASS test_gtts_success_keeps_mp3_path")


def _run():
    test_piper_fallback_writes_wav_when_gtts_offline()
    test_voice_returns_none_when_no_engine()
    test_gtts_success_keeps_mp3_path()


if __name__ == "__main__":
    _run()
