"""Configuration for Voice Cloner application."""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = Path.home() / ".cache" / "voice_cloner" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Audio settings
SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a"]
SAMPLE_RATE = 22050
MAX_AUDIO_DURATION_SEC = 30
MIN_AUDIO_DURATION_SEC = 3

# TTS model settings
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}

# UI settings
APP_TITLE = "🎙️ Voice Cloner Pro"
APP_ICON = "🎙️"
LAYOUT = "wide"
