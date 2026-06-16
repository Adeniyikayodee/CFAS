# CFAS: Community-Centred AI for Flood Resilience

Climate intelligence for the communities that stand closest to the water.

CFAS turns a satellite flood forecast into a spoken warning in a language a family already trusts, carried over the radio they already own. The forecast exists long before the water arrives. The work that remains is the bridge between that forecast and the household it protects, and this repository is that bridge, small enough to run on a single box at a village radio station.

## The idea

A forecast becomes protection once a family hears it, believes it, and still has time to act. So CFAS keeps the whole chain short and local:

1. **Read the landscape and the weather.** Three satellite layers fold into one number.
2. **Score the risk.** That number lands in three plain bands: LOW, MEDIUM, HIGH.
3. **Speak it plainly.** Gemma 3 drafts a calm advisory grounded in the sub-district, the hour, and one clear action.
4. **Carry it home.** NLLB-200 renders it into Twi, Hausa, Bambara and Yoruba, and the station broadcasts it.

The intelligence sits with the community, on a small device at the station, close to the people who live nearest the planet's fractures.

## How it works

The risk score follows one equation:

```
R(x) = alpha · P  +  beta · V  +  gamma · theta · mu(theta)
```

| Term | Meaning | Source |
| --- | --- | --- |
| `P` | rainfall forecast, looking ahead | **Google Weather API** forward forecast, up to 10 days (GFS and CHIRPS as fallbacks) |
| `V` | terrain, watercourses, settlement | **TESSERA** embeddings, the open foundation model from Cambridge |
| `theta` | surface soil moisture | NASA SMAP L4 |
| `mu(theta)` | saturation multiplier | lifts the soil-moisture term once the ground is already heavy with water |
| `alpha, beta, gamma` | calibration weights | yours to tune in `config.yaml` |

CFAS looks ahead. The rainfall term reads the Google Weather forecast, up to ten days out, so the score warns of a flood that is coming rather than scoring one that already passed. Set the window in `config.yaml` to today or later, add a `GOOGLE_WEATHER_KEY`, and each day in range carries its own forward forecast. When the key is away, the same term falls back to the Earth Engine GFS forecast, so the pipeline keeps predicting.

Once the band reaches MEDIUM, the message half begins:

```
W = Broadcast( NLLB-200( Gemma 3( R, L ) ), L )   for  L in {Twi, Hausa, Bambara, Yoruba}
```

Gemma 3 runs on the edge through **Cactus**, the on-device inference engine, so the drafting holds steady even as a storm reaches the nearest tower. The laptop and Raspberry Pi path falls back to a llama.cpp build of Gemma 3, and a plain template stands ready so the pipeline always speaks. The spoken audio comes from gTTS where there is a connection; where there is not, **Piper** voices the advisory fully offline on the same box, so a station cut off mid-storm still airs the warning. Set `PIPER_VOICES_DIR` to a folder of ONNX voices to enable it. Every translated line passes through a native speaker for review before it goes on air.

## A word on TESSERA

TESSERA gives `V` its eyes. It reads Sentinel-1 and Sentinel-2 into a 128-dimensional embedding at 10m resolution, and it covers land worldwide, Africa included. The 2024 embeddings span the globe today, so the Nigeria and Kenya pilots are ready to run, with earlier years arriving season by season. The embeddings are open under CC0, free to use and adapt, which keeps CFAS inspectable and rebuildable by the people who depend on it. TESSERA is annual, so it holds the slow picture, the terrain and settlement that shape where water gathers, while the Google Weather forecast and SMAP carry the fast layers that change day to day.

You pull TESSERA through the `geotessera` library. The pipeline fetches the tile under your community, averages the embedding, and maps it to a vulnerability score. Train a small linear probe on a handful of labelled sites and drop it in with `--vuln-head probe.npy` for a calibrated `V`. The shipped proxy keeps everything live until that probe exists.

## What's in here

```
cfas/
├── config.yaml        # where and when to watch, and how to weight the risk
├── .env.example       # your keys live here; copy to .env
├── requirements.txt
└── cfas/
    ├── risk.py        # satellite layers to a banded score (P, V, theta, mu, R)
    ├── forecast.py    # forward rainfall forecast from the Google Weather API
    ├── snapshot.py    # local cache of the layers, so the band computes offline
    ├── advisory.py    # score to spoken warning, plus call-in transcription
    ├── calibrate.py   # warnings measured against confirmed call-ins
    └── run.py         # the entrypoint that ties it together
└── tests/             # offline checks, including a real speech-to-text round trip
```

Two ends, clearly separated. Keys go in `.env`. Parameters go in `config.yaml`. The three modules stay steady underneath, ready for the next contributor to build on.

## Quick start

```bash
git clone <your-fork-url> cfas && cd cfas
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Your keys
cp .env.example .env        # add your Earth Engine project and model paths
earthengine authenticate    # one time on this machine

# 2. Your place and dates
$EDITOR config.yaml

# 3. Run it
python -m cfas.run
```

Want to see the message half first, before Earth Engine is wired up? Run the delivery pipeline on its own:

```bash
python -m cfas.run --dry-run
```

Each run writes a JSON record, a broadcast sheet, and an MP3 per language into `alerts/`.

## Running offline

The forecast arrives hours before the water, and that is the window CFAS works in. The analysis layers, the rainfall forecast, SMAP soil moisture, the TESSERA tile, all live in the cloud, but they reduce to a handful of small numbers per day plus one 128-dimensional embedding. So CFAS pulls them into a local **snapshot** while it has a connection, then computes the band, drafts, translates and broadcasts from that snapshot with no network at all. Cactus keeps the models on the box; the snapshot keeps the inputs on the box.

```bash
python -m cfas.run --snapshot     # online: pull the layers into OUTDIR/snapshot.json
python -m cfas.run --offline      # no network: band, draft, voice from the snapshot
```

The default run is offline-first: it uses a fresh snapshot if one is on disk, refreshes it live when it is stale and a connection is there, and falls back to the last snapshot with a loud warning when the line is down. The snapshot stores the **raw** inputs, not the finished `P`, `V` and `theta`, so you can re-tune `alpha`, `beta`, `gamma`, `prob_lean`, the rain triggers, even a trained `--vuln-head` probe, and re-band the same snapshot offline without fetching again.

Soil moisture and rainfall lose meaning as they age, so `--offline` refuses a snapshot older than `snapshot_max_age_hours` (default 24) unless you pass `--allow-stale`. TESSERA is annual, so its embedding ages slowly. The pattern in the field is a small cron that runs `--snapshot` whenever there is signal, leaving the station ready to broadcast through the storm that takes the tower down. See `snapshot.example.json` for the shape of the file.

## Listening back

A warning travels one way until the community speaks back. CFAS closes that loop with the same Cactus runtime that drafts the advisory.

When a listener calls the station with what they see, the river by the market, the road already under water, the goats moved in time, the station saves the voice note. Cactus transcribes it on the device, NLLB-200 carries it back to English, and the line joins a feedback ledger at `alerts/feedback.jsonl`. A native-speaker reviewer reads each entry and marks whether it confirms a flood, which turns the community's own words into the ground truth that calibrates the hit-rate.

```bash
# point it at a folder of call-in recordings, in the station's language
python -m cfas.run --listen ./callins --feedback-lang hausa
```

Each call-in lands as one JSON line, with the original transcript kept beside the English so the meaning stays close to how the caller said it. The audio stays on the device, which keeps the loop private by design.

## Measuring it

Trust is the currency of a warning system, and the way to earn it is to show the warnings hold up. Every run writes one band per day to `alerts/assessments.jsonl`, the complete record of what the model called. The call-in loop writes what the community saw. The calibrator joins the two and reports how well they agree.

```bash
python -m cfas.calibrate                   # reads ./alerts
python -m cfas.calibrate --alert-band HIGH  # the stricter view
```

It matches each confirmed call-in to the band for the same community and a nearby date, then sorts the result into four outcomes: a warning followed by a flood (a hit), a warning on a dry day (a false alarm), a quiet day that flooded (a miss), and a quiet day that stayed dry (a correct calm). From those it reports the three figures that decide whether a community keeps listening:

- **hit-rate (recall)**, the share of real floods the system warned about
- **precision**, the share of warnings that proved real
- **false-alarm rate**, the share of dry days that drew a warning

This is the measurement behind the projected 65 to 75 percent, and the dial you turn when you tune `alpha`, `beta`, `gamma`, and the band cutoffs in `config.yaml`. Raise the alert band and you trade a touch of recall for far fewer false alarms; the calibrator shows you the exact cost, so the choice stays yours.

## Languages

| Setting | Language | NLLB code (FLORES-200) |
| --- | --- | --- |
| `twi` | Twi | `aka_Latn` (covered as Akan) |
| `hausa` | Hausa | `hau_Latn` |
| `bambara` | Bambara | `bam_Latn` |
| `yoruba` | Yoruba | `yor_Latn` |

List the ones you need under `languages:` in `config.yaml`.

## Status and roadmap

A first field deployment is set for Q3 2026, with pilot districts in Nigeria and Kenya, running alongside community-radio partners who already hold the trust of local listeners. The projected hit-rate of 65 to 75 percent will be calibrated against real flood records as the deployment runs.

Next steps stay close to the ground:

- Train per-site vulnerability probes on TESSERA embeddings for a calibrated `V`.
- Add a real hydrological model with terrain and river networks in place of the current proxy.
- Benchmark TranslateGemma against NLLB-200 per language, with native speakers choosing the stronger model for each.
- Fold in community feedback so the warnings keep learning from the people who receive them.

## Limitations

CFAS is honest about what it is: a young system, still earning its trust. A few things to hold in mind before you lean on it.

- **It forecasts, it does not observe.** The band is a prediction of risk, not a measurement of a flood, and it can be wrong in both directions.
- **The vulnerability and hydrology are proxies.** Until a trained probe and a real river model land, `V` is a rough read of the terrain, not a calibrated one.
- **The 65 to 75 percent hit-rate is a projection.** It has not yet been proven against real flood records in the field.
- **Offline runs lean on the last snapshot.** A snapshot still needs a connection to refresh, and old soil-moisture and rainfall readings can mislead, which is why a stale run warns you.
- **Translations need a human.** Quality varies by language, every line passes a native speaker before air, and Twi currently rides on Akan.
- **Offline voices are thin for some languages.** Where Piper has no voice yet, the audio falls back to an English-accented one until a native voice exists.

In short, CFAS is a tool for the people at the station, not a replacement for official warnings or local judgement. It works best beside both.

## Tests

The suite runs offline, with no keys and no model downloads.

```bash
pip install pytest pocketsphinx   # pocketsphinx is optional, for the real ASR check
python -m pytest tests/            # or: python tests/test_listen.py
```

Two layers cover the call-in loop. One injects a scripted transcriber and walks the whole path from a voice note to the ledger to the calibrated metrics, fast and identical every run. The other runs genuine speech-to-text on a committed audio clip with pocketsphinx, so the ledger fills with real transcribed words; it skips cleanly when pocketsphinx is absent. The clip is the CMU Sphinx sample "go forward ten meters", which stands in for a listener call-in.

A third layer covers the flood processing itself. It feeds known layer values into the risk fusion and confirms the saturation multiplier, the score, and the LOW, MEDIUM, HIGH bands land where they should. A fourth checks the Google Weather forecast parser, confirming it sums the daytime and nighttime rainfall and scales the amount by the chance of rain. The live reads (Google Weather, SMAP, TESSERA, with GFS as the rainfall fallback) run against their services once your keys are in place.

## References

Each source maps to the part of the code that applies it.

1. Feng et al., *TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis*, arXiv:2506.20380 (2025), CVPR 2026. Vulnerability term `V` in `risk.py`, served through GeoTessera.
2. Brown et al., *AlphaEarth Foundations*, arXiv:2507.22291 (2025). The embedding approach TESSERA opens up, noted in `risk.py`.
3. Funk et al., *The climate hazards infrared precipitation with stations (CHIRPS)*, Scientific Data 2:150066 (2015). Observed-rainfall fallback for `P` in `risk.py`.
4. Google Maps Platform, *Weather API* (2026), developers.google.com/maps/documentation/weather. Forward rainfall forecast `P` in `forecast.py`.
4b. NOAA NCEP, *Global Forecast System (GFS)*. Fallback rainfall forecast `P` in `risk.py`.
5. Entekhabi et al., *The SMAP Mission*, Proc. IEEE 98(5):704 (2010). Soil-moisture term `theta` in `risk.py`.
6. Gemma Team, *Gemma 3 Technical Report*, arXiv:2503.19786 (2025). Advisory drafting in `advisory.py`.
7. Cactus Compute, github.com/cactus-compute/cactus (2025). On-device drafting and transcription runtime in `advisory.py`.
8. NLLB Team et al., *Scaling neural machine translation to 200 languages*, Nature 630:841 (2024). Translation and FLORES-200 codes in `advisory.py`.
9. Ojo et al., *AfroBench*, arXiv:2311.07978 (2024). The reason a native speaker reviews every line, in `advisory.py`.
10. Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*, arXiv:2212.04356 (2022). Local speech-to-text fallback in `advisory.py`.
11. Buytaert et al., *Citizen science in hydrology and water resources*, Front. Earth Sci. 2:26 (2014). The call-in feedback loop in `run.py`.
12. Jolliffe & Stephenson, *Forecast Verification: A Practitioner's Guide in Atmospheric Science*, 2nd ed., Wiley (2012). The contingency-table scoring in `calibrate.py`.
13. WMO & UNDRR, *Global Status of Multi-Hazard Early Warning Systems* (2023). The reach motivation behind the whole pipeline.

## License and citation

CFAS code: MIT. TESSERA embeddings and weights: CC0.

If TESSERA helps your work, please cite the paper above (`feng2025tesseratemporalembeddingssurface`).

---

Built by Kayode Adeniyi. Questions and collaboration welcome.
