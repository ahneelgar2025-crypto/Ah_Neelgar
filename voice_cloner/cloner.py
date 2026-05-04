"""Voice cloning engine using Coqui TTS XTTS v2."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import TTS_MODEL_NAME, SAMPLE_RATE, OUTPUT_DIR


class VoiceCloner:
    """Voice cloning engine wrapping Coqui TTS XTTS v2 model."""

    def __init__(self):
        self._tts = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    def load_model(self, progress_callback=None):
        """Load the XTTS v2 model."""
        if self._tts is not None:
            return

        from TTS.api import TTS

        if progress_callback:
            progress_callback("Downloading and loading XTTS v2 model...")

        self._tts = TTS(model_name=TTS_MODEL_NAME).to(self._device)

        if progress_callback:
            progress_callback("Model loaded successfully!")

    def clone_voice(
        self,
        text: str,
        speaker_wav_path: str,
        language: str = "en",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Clone a voice and synthesize speech.

        Args:
            text: Text to synthesize.
            speaker_wav_path: Path to reference speaker WAV file.
            language: Language code (e.g., 'en', 'es', 'fr').
            speed: Speech speed multiplier.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        if self._tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        self._tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_path,
            speed=speed,
        )

        audio, sr = sf.read(output_path)
        Path(output_path).unlink(missing_ok=True)

        return audio.astype(np.float32), sr

    def save_output(
        self, audio: np.ndarray, sr: int, filename: str = "cloned_voice.wav"
    ) -> Path:
        """Save generated audio to the output directory."""
        output_path = OUTPUT_DIR / filename
        sf.write(str(output_path), audio, sr)
        return output_path

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model": TTS_MODEL_NAME,
            "device": self._device,
            "loaded": self.is_loaded,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            ),
        }
