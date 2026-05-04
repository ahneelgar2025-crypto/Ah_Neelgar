# 🎙️ Voice Cloner Pro

Professional voice cloning application powered by **XTTS v2** (Coqui TTS) and **Streamlit**.

Clone any voice with just a short audio sample (3–30 seconds) and synthesize speech in 17+ languages.

---

## Features

- **Zero-Shot Voice Cloning** — Clone any voice with just 3–30 seconds of audio
- **17+ Languages** — English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Hindi, Korean, and more
- **Real-Time Audio Analysis** — Interactive waveform and mel spectrogram visualizations
- **Audio Preprocessing** — Automatic normalization, silence removal
- **Microphone Recording** — Record reference audio directly in the browser
- **Adjustable Speed** — Control speech speed (0.5×–2.0×)
- **Generation History** — Review and replay past generations
- **Download** — Export generated audio as WAV files

---

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)

### Installation

```bash
# Clone the repo
git clone https://github.com/ahneelgar2025-crypto/Ah_Neelgar.git
cd Ah_Neelgar/voice_cloner

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not installed)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

### Run the App

```bash
cd voice_cloner
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

### Step 1: Load the Model
Click **"🚀 Load Model"** in the sidebar. The XTTS v2 model (~1.8 GB) will be downloaded on first use.

### Step 2: Upload Reference Voice
Upload a clear voice recording (WAV, MP3, OGG, FLAC, M4A) or record directly using the microphone.

**Tips for best results:**
- Use **3–30 seconds** of clear speech
- Avoid background noise and music
- Ensure consistent volume
- Single speaker only

### Step 3: Generate Cloned Speech
1. Enter the text you want spoken
2. Select the target language
3. Adjust speech speed if needed
4. Click **"🎙️ Clone & Generate"**

### Step 4: Download
Download the generated audio using the download button.

---

## Project Structure

```
voice_cloner/
├── app.py              # Main Streamlit application
├── cloner.py           # Voice cloning engine (XTTS v2 wrapper)
├── audio_utils.py      # Audio processing utilities
├── config.py           # Configuration and constants
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit theme and settings
├── assets/             # Static assets
├── output/             # Generated audio files
└── README.md           # This file
```

---

## Technical Stack

| Component         | Technology                |
|-------------------|--------------------------|
| Voice Cloning     | XTTS v2 (Coqui TTS)      |
| Frontend          | Streamlit                 |
| Audio Processing  | librosa, pydub, soundfile |
| Visualization     | Plotly                    |
| Deep Learning     | PyTorch                   |

---

## System Requirements

| Requirement | Minimum        | Recommended      |
|-------------|----------------|------------------|
| RAM         | 8 GB           | 16 GB+           |
| Storage     | 4 GB free      | 8 GB free        |
| GPU         | Not required   | NVIDIA GPU (CUDA)|
| Python      | 3.9            | 3.10+            |

> **Note:** A CUDA-compatible GPU significantly speeds up voice cloning. CPU-only mode works but is slower.

---

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

---

## ⚠️ Ethical Use

This tool is intended for **educational and personal use only**.

- **Do not** clone voices without the speaker's explicit consent
- **Do not** use cloned voices for impersonation, fraud, or harassment
- **Always** disclose when audio is AI-generated
- Comply with all applicable laws and regulations regarding synthetic media

---

## License

This project is provided for educational purposes. The XTTS v2 model is subject to the [Coqui Public Model License](https://coqui.ai/cpml).
