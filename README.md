# Jev Voice — full-local fork (Laya, zero cloud)

> Fork of [kevinbadi/jev-voice](https://github.com/kevinbadi/jev-voice).
> The TypeSafe **Jev** cloud call is replaced by **Laya** running locally via
> [laya-mlx](https://github.com/mizorewww/laya-mlx) (Apple Silicon, ~150 ms).
> No API key, no telemetry, works offline. Trade-off (honest): Laya zero-shot
> is weak on this schema (argmax mostly right, confidences ~0.05, ~2.6 s per
> 18-question fan-out) — the fix is fine-tuning Laya on voice-command data
> (see roadmap below), not prompt tweaks.

Talk to your Mac. You speak, it opens apps, types, searches, scrolls, presses keys.

Everything runs locally: one **Laya** forward pass turns the transcript into a
typed action plus typed arguments in a single fan-out. Laya never generates
text; code produces candidate values and Laya *selects* (large option sets —
103 installed apps, 39 shortcuts — are fuzzy-shortlisted to `LAYA_MAX_OPTIONS`
in code first, because Laya's per-option token budget collapses past ~20).
Code owns execution.

```
mic ─► energy VAD ─► whisper.cpp (Metal, ~100 ms) ─► Laya (MLX, local) ─► macOS actions ─► `say`
```

## Setup (macOS, Apple Silicon)

```sh
cp .env.example .env
./scripts/setup.sh
```

No API key to add — `.env` only holds local settings (`LAYA_MODEL`,
`LAYA_MAX_OPTIONS`). First run downloads the Laya checkpoint (~800 MB, cached).

The script installs whisper-cpp + ffmpeg, downloads the model, syncs the Python
env, remaps **Caps Lock → F18** with `hidutil` (persisted by a LaunchAgent so it
survives reboots), installs a `jev` launcher in `~/.local/bin`, and opens the
three permission panes. Grant the terminal app you launch from (Cursor / Terminal /
iTerm) **Microphone**, **Accessibility** and **Input Monitoring**. If a permission
is missing at launch, Jev Voice prompts for it and waits.

Undo the Caps Lock remap any time: `./scripts/uninstall-capslock.sh`.

## Run

```sh
jev                                 # hands-free: "Alfred, open chrome" (or tap CAPS LOCK, then speak)
jev --hold                          # hold CAPS LOCK to talk, release to run; no wake word
jev --always-on                     # open mic, EVERY utterance is a command (no wake word)
jev --ptt                           # push-to-talk in the terminal: Enter start / Enter stop
jev --device "RØDE"                 # pick a mic (uv run python -m sounddevice)
jev --text "open chrome and go to youtube" --dry-run   # test routing, no mic
```

**Hands-free mode (default):** the mic stays open and whisper transcribes every
utterance locally (~100 ms, nothing leaves the machine). Only utterances that name
the assistant (`WAKE_WORDS` in `.env`, default Alfred / Jarvis) go to Jev. After a
command you have `FOLLOWUP_SECONDS` (8) to chain more without the name: "Alfred,
open chrome" … "go to youtube" … "scroll down". Saying just "Alfred" chimes and
arms the next utterance. A Caps Lock tap does the same.

**Caps Lock modes (`--hold`):** hold it while speaking (Tink = recording, Pop = sent). A
short tap (<250 ms) latches hands-free recording; tap again to send. Caps Lock no
longer toggles capitals while the remap is installed.

## What you can say

| Say | Does |
| --- | --- |
| "open cursor", "switch to chrome" | `open -a` the matching installed app (Jev picks from the real app list) |
| "go to youtube", "go to stripe dot com" | opens the site |
| "search youtube for lofi hip hop", "google best ramen near me" | site-specific search |
| "type hello world and hit enter" | types into the focused field, optional submit |
| "close this tab", "select all and copy", "undo", "go back", "reload" | ~45 keyboard shortcuts |
| "scroll down a lot", "go to the top" | real scroll-wheel events |
| "volume up", "mute", "pause the music", "next song" | system volume / media keys |
| "take a screenshot", "open my downloads", "lock the screen", "toggle dark mode" | misc |
| "open notes and type buy milk and press enter" | compound: Jev flags it, code splits it, each step runs in order |

## How the Jev layer works (`jev_voice/brain.py`)

One request per utterance with ~15 speculative questions evaluated in parallel:

- `action` — Choice over 13 action kinds.
- `app` — Choice over your installed apps (+ `none`); `site`, `engine`, `folder`,
  `shortcut`, `scroll_dir`, `volume_op`, `media_op`, `system_op` — Choices over
  closed sets whose keys are exactly what the executor accepts.
- `text` — Choice over **candidate spans** cut from the transcript by regex
  ("type X", "search for X", quoted text, whole utterance). Jev picks the one that
  is exactly the payload. This is the "select instead of generate" pattern.
- `submit`, `compound` — Nouls.

Code reads only the answers the chosen action needs. Plan confidence is the
minimum over the judgements used. Below `ACTION_MIN_CONFIDENCE` (0.35) it says
"not sure" instead of acting. Thresholds live in `jev_voice/config.py`.

## Latency (Mac mini M4, measured)

| Stage | Time |
| --- | --- |
| End-of-speech detection | 550 ms of silence (tune `VADConfig.end_silence_ms`) |
| whisper.cpp base.en | 80–130 ms |
| Jev fan-out | 170–420 ms |
| Execute + `say` | ~50–100 ms |

## Floating transcription pill

A small always-on-top bar at the top-center of the screen shows what whisper
heard, what Jev decided, and the result (gray idle · red listening · yellow
heard · blue thinking · green done · orange error). It never takes keyboard
focus. `OVERLAY=0` or `--no-overlay` hides it.

## Feedback

`FEEDBACK=ding` (default) plays a chime when an action completes and a low buzz
on failure. `FEEDBACK=voice` gives spoken replies from a posh butler persona
(`PERSONA=alfred`, or `cowboy`) using the best British voice installed, or
ElevenLabs if `ELEVENLABS_API_KEY` is set (phrases cached to disk, so repeats are
instant).

## Layout

```
jev_voice/
  main.py     loop, CLI, compound handling
  brain.py    Laya questions, fuzzy shortlist, candidate extraction, Plan
  actions.py  macOS execution (open, keystrokes, scroll, volume, media keys…)
  audio.py    mic + VAD endpointing
  stt.py      whisper-server client (localhost only)
  tts.py      macOS `say` (ElevenLabs only if ELEVENLABS_API_KEY is set)
  config.py   env / thresholds
  hotkey.py   Caps Lock (remapped to F18) global key tap
  overlay.py  floating transcription pill (AppKit)
  persona.py  butler / cowboy phrasing
scripts/
  setup.sh    one-shot install: deps, model, Caps Lock remap, launcher, permissions
```

## Roadmap: fine-tune Laya on voice commands

Measured on MacBook Air (laya-mlx, FP16): argmax correct on ~3/6 probe commands,
confidences ~0.03–0.35 (gate `ACTION_MIN_CONFIDENCE=0.35` blocks most), ~2.6 s
per 18-question fan-out. This matches Laya's model card: base checkpoints are
near-chance zero-shot (0.362 vs 0.318 random); the 0.766 figure needs
fine-tuning, and raw ECE is 0.466 until temperatures are refit. Next steps:

1. Collect training data: log `(utterance, apps, candidates, Jev-or-human labels)`
   for a few hundred real commands (distillation from the Jev API, or hand labels).
2. Fine-tune with the upstream notebook on Kaggle 2×T4 (free, ~4–5 h):
   `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in
   [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).
3. Convert the fine-tuned checkpoint to MLX, point `LAYA_MODEL` at it, refit
   temperatures per question type (ECE 0.466 → ~0.08), raise the gate back up.

## License

MIT
