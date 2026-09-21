from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
if TYPESAFE_API_KEY:
    import warnings

    warnings.warn(
        "TYPESAFE_API_KEY is ignored: this fork is full-local (laya-mlx), zero cloud.",
        DeprecationWarning,
        stacklevel=2,
    )
LAYA_MODEL = os.environ.get("LAYA_MODEL", "aac6fef/laya-mlx")
LAYA_MAX_OPTIONS = int(os.environ.get("LAYA_MAX_OPTIONS", "10"))
LAYA_BATCH_SIZE = int(os.environ.get("LAYA_BATCH_SIZE", "32"))

WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", ROOT / "models" / "ggml-base.en.bin"))
WHISPER_PORT = int(os.environ.get("WHISPER_PORT", "8178"))
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "6"))
# Initial prompt biases transcription toward our vocabulary (proper nouns like
# Claude/MDapp/WhatsApp are otherwise misheard: "Claude" -> "closed").
WHISPER_PROMPT = os.environ.get(
    "WHISPER_PROMPT",
    "Alfred, Jarvis. Open Chrome, Claude, ChatGPT, WhatsApp, MDapp, Safari, "
    "Cursor, Terminal, Finder, Notes, Messages. Go to YouTube. Scroll down. "
    "Volume up. Close this tab. Take a screenshot. Lock the screen.",
)

SAMPLE_RATE = 16000
TTS_VOICE = os.environ.get("TTS_VOICE", "Samantha")
TTS_RATE = int(os.environ.get("TTS_RATE", "210"))

# Confidence gates. Laya confidences are uncalibrated zero-shot (ECE ~0.47
# out-of-box), so deterministic code owns unambiguous patterns outright
# (conf 0.8-0.95) and the gate only needs to admit solid Laya plans while
# blocking shaky ones (e.g. a 0.13 website guess from a misheard word).
# Raise further after fine-tune + temperature fitting.
ACTION_MIN_CONFIDENCE = float(os.environ.get("ACTION_MIN_CONFIDENCE", "0.3"))
YES = 0.6
