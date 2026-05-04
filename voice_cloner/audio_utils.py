"""Audio processing utilities for Voice Cloner."""

import io
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from config import SAMPLE_RATE, MAX_AUDIO_DURATION_SEC, MIN_AUDIO_DURATION_SEC


def load_audio(file_bytes: bytes, file_name: str) -> tuple[np.ndarray, int]:
    """Load audio from uploaded file bytes."""
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    Path(tmp_path).unlink(missing_ok=True)
    return audio, sr


def save_audio_to_bytes(audio: np.ndarray, sr: int, fmt: str = "wav") -> bytes:
    """Convert audio array to bytes in the specified format."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=fmt)
    buf.seek(0)
    return buf.read()


def validate_audio_duration(audio: np.ndarray, sr: int) -> tuple[bool, str]:
    """Validate audio duration is within acceptable range."""
    duration = len(audio) / sr
    if duration < MIN_AUDIO_DURATION_SEC:
        return False, (
            f"Audio is too short ({duration:.1f}s). "
            f"Minimum {MIN_AUDIO_DURATION_SEC}s required for good voice cloning."
        )
    if duration > MAX_AUDIO_DURATION_SEC:
        return False, (
            f"Audio is too long ({duration:.1f}s). "
            f"Maximum {MAX_AUDIO_DURATION_SEC}s allowed. Trimming recommended."
        )
    return True, f"Audio duration: {duration:.1f}s"


def trim_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Trim audio to the specified time range (in seconds)."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def remove_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Remove leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def get_audio_info(audio: np.ndarray, sr: int) -> dict:
    """Get audio metadata."""
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "samples": len(audio),
        "rms_level": round(float(rms), 4),
        "peak_level": round(float(peak), 4),
        "channels": 1,
    }


def audio_bytes_to_wav(file_bytes: bytes, file_name: str) -> str:
    """Convert any audio format to WAV and return the temp file path."""
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(file_bytes)
        tmp_in_path = tmp_in.name

    audio_segment = AudioSegment.from_file(tmp_in_path)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(SAMPLE_RATE)

    wav_path = tmp_in_path.rsplit(".", 1)[0] + "_converted.wav"
    audio_segment.export(wav_path, format="wav")

    Path(tmp_in_path).unlink(missing_ok=True)
    return wav_path


def compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel spectrogram for visualization."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
