"""
Voice Cloner Pro - Professional Voice Cloning Application
Built with Streamlit and Coqui TTS XTTS v2.
"""

import io
import time
import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from audio_utils import (
    audio_bytes_to_wav,
    compute_mel_spectrogram,
    get_audio_info,
    load_audio,
    normalize_audio,
    remove_silence,
    save_audio_to_bytes,
    validate_audio_duration,
)
from cloner import VoiceCloner
from config import (
    APP_ICON,
    APP_TITLE,
    LAYOUT,
    MAX_AUDIO_DURATION_SEC,
    MIN_AUDIO_DURATION_SEC,
    SUPPORTED_FORMATS,
    SUPPORTED_LANGUAGES,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=LAYOUT)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FED330 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #2D3348;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FF6B6B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
    }
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2D3348;
        border-radius: 10px;
        padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    .footer {
        text-align: center;
        color: #555;
        padding: 2rem 0 1rem;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ────────────────────────────────────────────────────────────


def init_session_state():
    defaults = {
        "cloner": None,
        "model_loaded": False,
        "reference_audio": None,
        "reference_path": None,
        "generated_audio": None,
        "generated_sr": None,
        "generation_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Helper functions ─────────────────────────────────────────────────────────


def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform") -> go.Figure:
    """Create an interactive waveform plot."""
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    # Downsample for plotting if too many points
    max_points = 5000
    if len(audio) > max_points:
        step = len(audio) // max_points
        time_axis = time_axis[::step]
        audio = audio[::step]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=audio,
            mode="lines",
            line=dict(color="#FF6B6B", width=1),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.1)",
            name="Amplitude",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_spectrogram(
    audio: np.ndarray, sr: int, title: str = "Mel Spectrogram"
) -> go.Figure:
    """Create an interactive mel spectrogram plot."""
    mel_spec = compute_mel_spectrogram(audio, sr)

    fig = go.Figure(
        data=go.Heatmap(
            z=mel_spec,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Time Frames",
        yaxis_title="Mel Bands",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Voice Cloner Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Professional voice cloning powered by XTTS v2 &mdash; "
    "clone any voice with just a short audio sample"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Model status
    st.markdown("### 🤖 Model Status")
    if st.session_state.model_loaded:
        st.success("XTTS v2 Model Loaded")
        info = st.session_state.cloner.get_model_info()
        st.caption(f"Device: **{info['device'].upper()}**")
        if info["gpu_available"]:
            st.caption(f"GPU: **{info['gpu_name']}**")
    else:
        st.warning("Model not loaded")
        if st.button("🚀 Load Model", use_container_width=True):
            with st.spinner("Loading XTTS v2 model... This may take a few minutes."):
                cloner = VoiceCloner()
                cloner.load_model()
                st.session_state.cloner = cloner
                st.session_state.model_loaded = True
            st.rerun()

    st.divider()

    # Synthesis settings
    st.markdown("### 🎛️ Synthesis Settings")

    language = st.selectbox(
        "Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        help="Select the language for the synthesized speech.",
    )
    lang_code = SUPPORTED_LANGUAGES[language]

    speed = st.slider(
        "Speech Speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust the speed of the generated speech.",
    )

    st.divider()

    # Audio preprocessing
    st.markdown("### 🔧 Preprocessing")

    do_normalize = st.checkbox(
        "Normalize Audio", value=True, help="Normalize the reference audio volume."
    )

    do_trim_silence = st.checkbox(
        "Remove Silence",
        value=True,
        help="Remove leading and trailing silence from the reference audio.",
    )

    st.divider()

    # Generation history
    st.markdown("### 📜 History")
    if st.session_state.generation_history:
        for i, entry in enumerate(reversed(st.session_state.generation_history[-5:])):
            with st.expander(f"#{len(st.session_state.generation_history) - i}: {entry['text'][:30]}..."):
                st.caption(f"Language: {entry['language']}")
                st.caption(f"Speed: {entry['speed']}x")
                st.caption(f"Time: {entry['timestamp']}")
                st.audio(entry["audio_bytes"], format="audio/wav")
    else:
        st.caption("No generations yet.")


# ── Main content ─────────────────────────────────────────────────────────────

tab_clone, tab_compare, tab_about = st.tabs(
    ["🎙️ Voice Cloning", "📊 Audio Analysis", "ℹ️ About"]
)

# ── Tab 1: Voice Cloning ─────────────────────────────────────────────────────

with tab_clone:
    col_upload, col_generate = st.columns([1, 1], gap="large")

    # -- Left column: Upload reference voice --
    with col_upload:
        st.markdown(
            '<span class="step-badge">Step 1</span>', unsafe_allow_html=True
        )
        st.markdown("#### Upload Reference Voice")
        st.caption(
            f"Upload a clear voice sample ({MIN_AUDIO_DURATION_SEC}-{MAX_AUDIO_DURATION_SEC}s). "
            "Supported: WAV, MP3, OGG, FLAC, M4A"
        )

        uploaded_file = st.file_uploader(
            "Upload voice sample",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed",
            help="Upload a clear voice recording for cloning.",
        )

        # Microphone recording
        st.markdown("**— or record directly —**")
        try:
            from audio_recorder_streamlit import audio_recorder

            recorded_bytes = audio_recorder(
                text="Click to record",
                recording_color="#FF6B6B",
                neutral_color="#2D3348",
                icon_size="2x",
                pause_threshold=3.0,
            )
        except ImportError:
            recorded_bytes = None
            st.info("Install `audio-recorder-streamlit` to enable microphone recording.")

        # Process uploaded or recorded audio
        audio_source = None
        audio_name = None

        if uploaded_file is not None:
            audio_source = uploaded_file.getvalue()
            audio_name = uploaded_file.name
        elif recorded_bytes is not None:
            audio_source = recorded_bytes
            audio_name = "recorded_audio.wav"

        if audio_source is not None:
            audio, sr = load_audio(audio_source, audio_name)

            # Preprocessing
            if do_trim_silence:
                audio = remove_silence(audio, sr)
            if do_normalize:
                audio = normalize_audio(audio)

            # Validate
            is_valid, msg = validate_audio_duration(audio, sr)
            if is_valid:
                st.success(msg)
            else:
                st.error(msg)

            # Audio info
            info = get_audio_info(audio, sr)
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Duration", f"{info['duration_sec']}s")
            with col_m2:
                st.metric("Sample Rate", f"{info['sample_rate']} Hz")
            with col_m3:
                st.metric("Peak Level", f"{info['peak_level']:.2f}")

            # Playback
            st.audio(audio_source, format="audio/wav")

            # Waveform
            fig_wave = plot_waveform(audio, sr, "Reference Voice Waveform")
            st.plotly_chart(fig_wave, use_container_width=True)

            # Save reference for cloning
            wav_path = audio_bytes_to_wav(audio_source, audio_name)
            st.session_state.reference_audio = audio
            st.session_state.reference_path = wav_path
        else:
            st.info("👆 Upload a voice sample or record one to get started.")

    # -- Right column: Generate cloned speech --
    with col_generate:
        st.markdown(
            '<span class="step-badge">Step 2</span>', unsafe_allow_html=True
        )
        st.markdown("#### Generate Cloned Speech")

        text_input = st.text_area(
            "Enter text to synthesize",
            placeholder=(
                "Type or paste the text you want spoken in the cloned voice...\n\n"
                "Example: Hello, this is my cloned voice speaking! "
                "I can say anything you want me to say."
            ),
            height=180,
            help="Enter the text that will be spoken in the cloned voice.",
        )

        # Character count
        if text_input:
            char_count = len(text_input)
            color = "#4CAF50" if char_count <= 500 else "#FF9800" if char_count <= 1000 else "#F44336"
            st.caption(
                f"Characters: <span style='color:{color}'>{char_count}</span>",
                unsafe_allow_html=True,
            )

        # Generate button
        can_generate = (
            st.session_state.model_loaded
            and st.session_state.reference_path is not None
            and text_input
            and len(text_input.strip()) > 0
        )

        if not st.session_state.model_loaded:
            st.warning("⚠️ Load the model first (sidebar)")
        elif st.session_state.reference_path is None:
            st.info("⬅️ Upload a reference voice first")

        generate_clicked = st.button(
            "🎙️ Clone & Generate",
            use_container_width=True,
            disabled=not can_generate,
            type="primary",
        )

        if generate_clicked and can_generate:
            with st.spinner("🔄 Cloning voice and generating speech..."):
                start_time = time.time()

                try:
                    audio_out, sr_out = st.session_state.cloner.clone_voice(
                        text=text_input,
                        speaker_wav_path=st.session_state.reference_path,
                        language=lang_code,
                        speed=speed,
                    )

                    elapsed = time.time() - start_time

                    st.session_state.generated_audio = audio_out
                    st.session_state.generated_sr = sr_out

                    # Success metrics
                    st.success(f"Voice cloned successfully in {elapsed:.1f}s!")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric("Generation Time", f"{elapsed:.1f}s")
                    with col_t2:
                        st.metric(
                            "Audio Length",
                            f"{len(audio_out) / sr_out:.1f}s",
                        )

                    # Playback
                    audio_bytes = save_audio_to_bytes(audio_out, sr_out)
                    st.markdown("##### 🔊 Generated Audio")
                    st.audio(audio_bytes, format="audio/wav")

                    # Waveform
                    fig_gen = plot_waveform(audio_out, sr_out, "Generated Voice Waveform")
                    st.plotly_chart(fig_gen, use_container_width=True)

                    # Download
                    st.download_button(
                        label="⬇️ Download Generated Audio",
                        data=audio_bytes,
                        file_name=f"cloned_voice_{int(time.time())}.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

                    # Add to history
                    st.session_state.generation_history.append(
                        {
                            "text": text_input,
                            "language": language,
                            "speed": speed,
                            "timestamp": datetime.datetime.now().strftime(
                                "%H:%M:%S"
                            ),
                            "audio_bytes": audio_bytes,
                            "duration": len(audio_out) / sr_out,
                        }
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

# ── Tab 2: Audio Analysis ────────────────────────────────────────────────────

with tab_compare:
    st.markdown("#### 📊 Audio Comparison & Analysis")
    st.caption("Compare reference voice with generated output")

    if (
        st.session_state.reference_audio is not None
        and st.session_state.generated_audio is not None
    ):
        col_ref, col_gen = st.columns(2)

        with col_ref:
            st.markdown("##### Reference Voice")
            ref_audio = st.session_state.reference_audio
            ref_sr = 22050

            fig_ref_wave = plot_waveform(ref_audio, ref_sr, "Reference Waveform")
            st.plotly_chart(fig_ref_wave, use_container_width=True)

            fig_ref_spec = plot_spectrogram(ref_audio, ref_sr, "Reference Spectrogram")
            st.plotly_chart(fig_ref_spec, use_container_width=True)

            ref_info = get_audio_info(ref_audio, ref_sr)
            st.json(ref_info)

        with col_gen:
            st.markdown("##### Generated Voice")
            gen_audio = st.session_state.generated_audio
            gen_sr = st.session_state.generated_sr

            fig_gen_wave = plot_waveform(gen_audio, gen_sr, "Generated Waveform")
            st.plotly_chart(fig_gen_wave, use_container_width=True)

            fig_gen_spec = plot_spectrogram(gen_audio, gen_sr, "Generated Spectrogram")
            st.plotly_chart(fig_gen_spec, use_container_width=True)

            gen_info = get_audio_info(gen_audio, gen_sr)
            st.json(gen_info)
    else:
        st.info(
            "Generate a voice clone first to see the comparison. "
            "Go to the **Voice Cloning** tab to get started."
        )

# ── Tab 3: About ─────────────────────────────────────────────────────────────

with tab_about:
    st.markdown(
        """
    #### About Voice Cloner Pro

    **Voice Cloner Pro** is a professional-grade voice cloning application
    powered by **XTTS v2** from Coqui TTS.

    ---

    ##### Features
    - **Zero-shot voice cloning** — clone any voice with just 3-30 seconds of audio
    - **Multi-language support** — 17+ languages including English, Spanish, French,
      German, Chinese, Japanese, Hindi, and more
    - **Real-time audio analysis** — waveform and mel spectrogram visualizations
    - **Audio preprocessing** — automatic normalization, silence removal
    - **Adjustable parameters** — control speech speed
    - **Microphone recording** — record reference audio directly in the browser
    - **Download & history** — download generated audio and review past generations

    ---

    ##### How It Works
    1. **Upload** a clear voice sample (3-30 seconds)
    2. **Enter** the text you want spoken in the cloned voice
    3. **Generate** — the AI will clone the voice and synthesize speech
    4. **Download** your cloned audio

    ---

    ##### Technical Details
    | Component | Technology |
    |-----------|-----------|
    | Voice Cloning | XTTS v2 (Coqui TTS) |
    | Frontend | Streamlit |
    | Audio Processing | librosa, pydub, soundfile |
    | Visualization | Plotly |
    | Deep Learning | PyTorch |

    ---

    ##### Tips for Best Results
    - Use **clear, noise-free** recordings
    - Ensure the reference audio is **3-30 seconds** long
    - Use recordings with **consistent volume** and **no background music**
    - Longer reference audio generally gives **better voice matching**
    - Keep synthesized text **under 500 characters** per generation for best quality

    ---

    ##### ⚠️ Ethical Use
    This tool is for **educational and personal use only**.
    - Do **not** clone voices without the speaker's consent
    - Do **not** use cloned voices for impersonation, fraud, or harassment
    - Always disclose when audio is AI-generated
    """
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div class="footer">'
    "Voice Cloner Pro &mdash; Built with Streamlit &amp; Coqui TTS XTTS v2"
    "</div>",
    unsafe_allow_html=True,
)
