"""A risk score turned into a spoken, local-language warning, and listener voice
notes turned back into text for a feedback loop.

    draft       Gemma 3              Gemma Team, "Gemma 3 Technical Report,"
                                     arXiv:2503.19786 (2025)
    runtime     Cactus on device     Cactus Compute, github.com/cactus-compute/
                                     cactus (2025) -> llama.cpp -> template
    translate   NLLB-200             NLLB Team et al., "Scaling neural machine
                                     translation to 200 languages," Nature
                                     630:841 (2024); FLORES-200 codes from the
                                     same work
    voice       gTTS -> Piper        Google TTS for radio when there is a network
                                     (online), with Piper voicing it fully on the
                                     device when there is not (rhasspy/piper)
    transcribe  Cactus STT           on device, with Whisper as the local stand-in
                                     (Radford et al., "Robust Speech Recognition
                                     via Large-Scale Weak Supervision,"
                                     arXiv:2212.04356, 2022)

Cactus carries the drafting locally at the station, so the broadcast holds steady
even as a storm reaches the nearest tower. NLLB-200 renders the advisory in each
target language, and a native speaker reviews every line before it airs, since
quality across African languages varies model to model (Ojo et al., "AfroBench,"
arXiv:2311.07978, 2024).

The same Cactus runtime also listens. When a community member calls the station
with what they see on the ground, Cactus transcribes the voice note on the device,
NLLB-200 carries it back to English, and the line joins a feedback ledger the team
calibrates against real flood records (community feedback as climate infrastructure:
Buytaert et al., Front. Earth Sci. 2:26, 2014).
"""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

# Target languages with their FLORES-200 codes (NLLB Team et al., Nature 630:841,
# 2024). Twi rides on Akan (aka_Latn), the closest entry NLLB covers. The last two
# fields hold the gTTS voice and the Whisper/ISO hint for transcription.
Lang = namedtuple("Lang", "label flores tts iso")
LANGS = {
    "twi":     Lang("Twi",     "aka_Latn", "en", None),
    "hausa":   Lang("Hausa",   "hau_Latn", "ha", "ha"),
    "bambara": Lang("Bambara", "bam_Latn", "en", None),
    "yoruba":  Lang("Yoruba",  "yor_Latn", "yo", "yo"),
}
SOURCE = "eng_Latn"

# Each band points to one plain action a household can take, the CFAS idea of
# handing people a decision rather than a probability (Adeniyi, CFAS, 2026).
ACTIONS = {
    "LOW":    "Keep listening to this station and watch the river.",
    "MEDIUM": "Move animals and stored grain to higher ground today.",
    "HIGH":   "Move people, animals and food to higher ground now, ahead of the water.",
}


class Advisor:
    """Drafts in English with Gemma 3, then translates with NLLB-200."""

    def __init__(self, *, cactus_url=None, gemma_model="gemma-3-it.gguf",
                 nllb_model="facebook/nllb-200-distilled-600M", stt_model="base",
                 transcriber=None, piper_voices=None):
        self.cactus_url = cactus_url or os.environ.get("CACTUS_URL")
        self.gemma_model = gemma_model
        self.nllb_model = nllb_model
        self.stt_model = stt_model or os.environ.get("STT_MODEL", "base")
        # transcriber lets a deployment plug in any speech-to-text it prefers,
        # taking (audio_path, lang_hint) and returning text. The Cactus and Whisper
        # path stays the default when this is left as the sentinel None.
        self.transcriber = transcriber
        # piper_voices is a folder of offline ONNX voices for the gTTS fallback,
        # so the station still airs the warning with no network. None disables it.
        self.piper_voices = piper_voices or os.environ.get("PIPER_VOICES_DIR")
        self._llama = None
        self._nllb = None
        self._tok = None
        self._whisper = None
        self._piper = {}

    # -- draft (Gemma 3) -----------------------------------------------------
    # Gemma 3 writes the advisory in plain English, grounded in the place, the hour
    # and one action (Gemma Team, arXiv:2503.19786, 2025). It runs on the edge through
    # Cactus (Cactus Compute, 2025), so a station keeps drafting while the line is
    # down. A llama.cpp build serves as the laptop and Raspberry Pi path, and a plain
    # template stands ready so the pipeline always speaks.
    def draft(self, ctx: dict) -> str:
        prompt = (
            "You are a trusted community-radio reporter. Write a short flood advisory "
            "in plain English for rural listeners, about 45 seconds of airtime, three or "
            "four sentences. Ground it in the named sub-district, the expected time, and "
            "one clear action. Keep a calm, steady tone, name the local landmark, and use "
            "local time.\n"
            "Output only the words the presenter reads aloud, as one short plain-prose "
            "paragraph. Do not add any preamble, sign-off about the text itself, sound "
            "effects, music cues, stage directions, parentheses, asterisks, markdown, "
            "headings, or surrounding quotation marks. Begin directly with the advisory.\n\n"
            f"Place: {ctx['community']}, {ctx['subcounty']}, {ctx['county']}, {ctx['country']}\n"
            f"Risk band: {ctx['band']}\n"
            f"Expected time: {ctx['hour']} on {ctx['date']}\n"
            f"Rainfall: {ctx['rain_mm']} mm. Ground reads as "
            f"{'saturated' if ctx['soil'] > 0.30 else 'moderately wet'}.\n"
            f"Landmark: {ctx['landmark']}\n"
            f"Action: {ctx['action']}\n"
        )
        for backend in (self._draft_cactus, self._draft_llama):
            text = self._clean(backend(prompt))
            if text:
                return text
        return self._template(ctx)

    @staticmethod
    def _clean(text):
        # A small model still slips in preambles ("Okay, here's the text:"),
        # music cues and stage directions ("(Sound of music fades)"), markdown
        # and wrapping quotes, even when the prompt forbids them. Strip those so
        # what reaches the broadcast sheet is only the words the presenter reads.
        if not text:
            return text
        import re
        kept = []
        for ln in text.splitlines():
            s = ln.strip().strip("*").strip()
            if not s:
                continue
            if re.fullmatch(r"\(.*\)", s):  # whole-line stage direction / cue
                continue
            if re.match(r"(?i)^(okay|sure|here'?s|here is|below is|advisory text|"
                        r"here you go)\b[^.!?]*:\s*$", s):  # meta preamble line
                continue
            kept.append(s)
        out = " ".join(kept).replace("*", "").strip()
        return out.strip("\"“”‘’«» ").strip()

    def _draft_cactus(self, prompt):
        if not self.cactus_url:
            return None
        try:
            import requests
            r = requests.post(f"{self.cactus_url}/v1/chat/completions", timeout=60, json={
                "model": self.gemma_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5, "max_tokens": 220})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _draft_llama(self, prompt):
        try:
            if self._llama is None:
                from llama_cpp import Llama
                if not os.path.exists(self.gemma_model):
                    return None
                self._llama = Llama(model_path=self.gemma_model, n_ctx=2048,
                                    n_threads=os.cpu_count() or 4, verbose=False)
            out = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=220)
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    @staticmethod
    def _template(ctx):
        # The ground phrase tracks the real soil reading, so the advisory never
        # states a wetness it cannot back up.
        ground = ("the ground is already soaked" if ctx["soil"] > 0.30
                  else "the ground can fill fast")
        return (
            f"Good day, listeners in {ctx['community']}. Our flood watch for {ctx['subcounty']} "
            f"reads {ctx['band']} for {ctx['hour']} on {ctx['date']}. Heavy rain is on the way "
            f"and {ground}, so water can rise quickly by the river. "
            f"{ctx['action']} Head for {ctx['landmark']} if you move, and stay with this station "
            f"for updates.")

    # -- translate (NLLB-200, either direction) ------------------------------
    # NLLB-200 carries the advisory into each target language, and the same model
    # carries listener call-ins back to English (NLLB Team et al., Nature 630:841,
    # 2024). The forced BOS token selects the FLORES-200 target. The placeholder on
    # the exception path keeps a clear marker for the native speaker, who has the
    # final say since quality runs unevenly across these languages (Ojo et al.,
    # "AfroBench," arXiv:2311.07978, 2024).
    def translate(self, text: str, tgt: str, src: str = SOURCE) -> str:
        if not text or tgt == src:
            return text
        try:
            self._ensure_nllb()
            self._tok.src_lang = src
            enc = self._tok(text, return_tensors="pt", truncation=True, max_length=400)
            gen = self._nllb.generate(**enc, max_length=400,
                                      forced_bos_token_id=self._tok.convert_tokens_to_ids(tgt))
            return self._tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        except Exception:
            if tgt == SOURCE:
                return text
            label = next((v.label for v in LANGS.values() if v.flores == tgt), tgt)
            return f"[{label}, awaiting native-speaker review] {text}"

    def _ensure_nllb(self):
        if self._nllb is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.nllb_model)
            self._nllb = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_model)

    # -- transcribe (Cactus STT, for the call-in feedback loop) --------------
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> str:
        """A listener's voice note to text. Cactus runs speech-to-text on the device
        (Cactus Compute, 2025), and a local faster-whisper build stands in when
        Cactus is away (Whisper: Radford et al., arXiv:2212.04356, 2022). The audio
        stays on the device, which keeps the loop private by design."""
        if self.transcriber is not None:
            return self.transcriber(audio_path, lang_hint) or ""
        text = self._stt_cactus(audio_path, lang_hint)
        return text if text is not None else (self._stt_local(audio_path, lang_hint) or "")

    def _stt_cactus(self, path, lang_hint):
        if not self.cactus_url:
            return None
        try:
            import requests
            with open(path, "rb") as fh:
                data = {"model": self.stt_model}
                if lang_hint:
                    data["language"] = lang_hint
                r = requests.post(f"{self.cactus_url}/v1/audio/transcriptions",
                                  files={"file": (os.path.basename(path), fh)},
                                  data=data, timeout=120)
            r.raise_for_status()
            return r.json().get("text", "").strip()
        except Exception:
            return None

    def _stt_local(self, path, lang_hint):
        try:
            from faster_whisper import WhisperModel
            if self._whisper is None:
                self._whisper = WhisperModel(self.stt_model, device="cpu", compute_type="int8")
            segments, _ = self._whisper.transcribe(path, language=lang_hint)
            return " ".join(s.text for s in segments).strip()
        except Exception:
            return None

    # -- voice: gTTS online, Piper offline -----------------------------------
    # The broadcast must still find a voice when the tower is down, so synthesis
    # degrades the way drafting and transcription do: try the networked path first
    # for its quality, then an on-device engine that needs no connection. gTTS
    # (Google Translate TTS) gives the better voices but needs the network; Piper
    # runs fully offline on the edge box (Raspberry-Pi class), so a station cut off
    # mid-storm can still put the warning on air. Returns the path of the audio
    # actually written (gTTS writes .mp3, Piper writes .wav), or None when neither
    # engine could render, so the caller knows what reached the broadcast folder.
    def voice(self, text: str, gtts_code: str, path: str) -> str | None:
        return self._voice_gtts(text, gtts_code, path) or self._voice_piper(text, gtts_code, path)

    @staticmethod
    def _voice_gtts(text, gtts_code, path):
        # Online path. Falls back to an English voice when the target language has
        # no gTTS voice, then gives up so Piper can take the offline turn.
        try:
            from gtts import gTTS
        except Exception:
            return None
        for lang in (gtts_code, "en"):
            try:
                gTTS(text=text, lang=lang, slow=False).save(path)
                return path
            except Exception:
                continue
        return None

    def _voice_piper(self, text, gtts_code, path):
        # Offline path. Piper synthesises on the device with an ONNX voice, so the
        # advisory still airs with no network (Piper, github.com/rhasspy/piper).
        # Drop voice models named <gtts_code>.onnx (e.g. ha.onnx, yo.onnx, en.onnx,
        # where en covers Twi and Bambara) into PIPER_VOICES_DIR; default.onnx, then
        # any voice present, stand in when a language-specific one is missing. Piper
        # writes 16-bit PCM WAV, so the output keeps a .wav name rather than label
        # itself .mp3 and hand the radio software a file that lies about its format.
        if not self.piper_voices:
            return None
        try:
            model = self._piper_voice(gtts_code)
            if model is None:
                return None
            import wave
            out = str(Path(path).with_suffix(".wav"))
            with wave.open(out, "wb") as wav:
                # Piper's API has shifted across releases; prefer the explicit
                # WAV writer (>= 1.2) and fall back to the older synthesize(text, wav).
                if hasattr(model, "synthesize_wav"):
                    model.synthesize_wav(text, wav)
                else:
                    model.synthesize(text, wav)
            return out
        except Exception:
            return None

    def _piper_voice(self, gtts_code):
        # Load and cache one Piper voice per language code, the way the NLLB and
        # Whisper models are loaded once and reused across calls. A cached None
        # records "no voice available" so a missing model is not retried each call.
        if gtts_code in self._piper:
            return self._piper[gtts_code]
        voice = None
        try:
            from piper import PiperVoice
            d = Path(self.piper_voices)
            onnx = (next((c for c in (d / f"{gtts_code}.onnx", d / "default.onnx")
                          if c.exists()), None)
                    or next(iter(sorted(d.glob("*.onnx"))), None))
            if onnx is not None:
                voice = PiperVoice.load(str(onnx))
        except Exception:
            voice = None
        self._piper[gtts_code] = voice
        return voice
