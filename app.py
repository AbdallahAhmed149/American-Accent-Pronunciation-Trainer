import asyncio
import difflib
import html
import os
import re
import tempfile
import edge_tts
import eng_to_ipa as ipa
import streamlit as st
from faster_whisper import WhisperModel
import io
import wave
import numpy as np

VOICES = {
    "Andrew (Male - Conversational)": "en-US-AndrewMultilingualNeural",
    "Ava (Female - Conversational)": "en-US-AvaMultilingualNeural",
    "Brian (Male - Clear/Warm)": "en-US-BrianMultilingualNeural",
    "Emma (Female - Clear/Warm)": "en-US-EmmaMultilingualNeural",
    "Christopher (Male - Deep/News)": "en-US-ChristopherNeural",
}


@st.cache_resource
def load_whisper_model() -> WhisperModel:
    return WhisperModel("base.en", device="cpu", compute_type="int8")


async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


@st.cache_data(show_spinner=False)
def get_native_audio(text: str, voice: str, rate: str = "+0%") -> bytes:
    return asyncio.run(_synthesize(text, voice, rate))


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def evaluate_pronunciation(target: str, transcript: str):
    target_words = normalize_words(target)
    spoken_words = normalize_words(transcript)

    if not target_words:
        return 0.0, 0, 0, "", []

    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
    parts = []
    missed_words = []
    matched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in target_words[i1:i2]:
                parts.append(f'<span style="color:#22c55e;font-weight:600;">{html.escape(w)}</span>')
                matched += 1
        elif tag == "replace":
            expected = html.escape(" ".join(target_words[i1:i2]))
            heard = html.escape(" ".join(spoken_words[j1:j2]))
            parts.append(
                f'<span style="color:#ef4444;font-weight:600;text-decoration:underline;">{expected}</span> '
                f'<span style="color:#eab308;font-size:0.85em;">(heard: {heard})</span>'
            )
            missed_words.extend(target_words[i1:i2])
        elif tag == "delete":
            for w in target_words[i1:i2]:
                parts.append(f'<span style="color:#ef4444;font-weight:600;">[{html.escape(w)}]</span>')
                missed_words.append(w)
        elif tag == "insert":
            extra = html.escape(" ".join(spoken_words[j1:j2]))
            parts.append(f'<span style="color:#eab308;font-size:0.85em;">(+{extra})</span>')

    accuracy = (matched / len(target_words)) * 100
    unique_missed = list(dict.fromkeys(missed_words))
    return accuracy, matched, len(target_words), " ".join(parts), unique_missed


def transcribe_wav_bytes(model: WhisperModel, wav_bytes: bytes) -> str:
    """Decode WAV bytes directly to a 16 kHz float32 NumPy array (bypasses PyAV)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw_frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz (Whisper's native sample rate)
    if framerate != 16000:
        target_len = int(len(audio) * 16000 / framerate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


# --- UI Setup ---
st.set_page_config(page_title="American Accent Trainer", layout="centered")
st.title("American Accent & Pronunciation Trainer")

with st.sidebar:
    st.header("Settings")
    voice_label = st.selectbox("Native Voice", list(VOICES.keys()))
    voice_id = VOICES[voice_label]
    slow_rate = st.slider("Shadowing Slow Speed (%)", min_value=-40, max_value=-10, value=-25, step=5)

sentence = st.text_area(
    "Enter a word or sentence to practice:",
    placeholder="I'm gonna grab a cup of water before we head out.",
    height=90,
).strip()

if sentence:
    # 1. American IPA Breakdown
    st.markdown(f"**American IPA:** `/{ipa.convert(sentence)}/`")

    # 2. Native Audio Players (Slow + Normal side-by-side)
    with st.spinner("Generating American native audio..."):
        slow_audio = get_native_audio(sentence, voice_id, rate=f"{slow_rate}%")
        normal_audio = get_native_audio(sentence, voice_id, rate="+0%")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Slow ({slow_rate}%) — Shadowing")
        st.audio(slow_audio, format="audio/mp3")
    with col2:
        st.caption("Normal (1.0x) — Native Speed")
        st.audio(normal_audio, format="audio/mp3")

    st.divider()

    # 3. Microphone Recording & Whisper Scoring
    st.subheader("Record & Score Your Pronunciation")
    recorded_audio = st.audio_input("Click the microphone to record")

    if recorded_audio is not None:
        wav_bytes = recorded_audio.getvalue()

        with st.spinner("Transcribing with Whisper (base.en)..."):
            model = load_whisper_model()
            transcript = transcribe_wav_bytes(model, wav_bytes)

        accuracy, matched, total, diff_html, missed_words = evaluate_pronunciation(sentence, transcript)

        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.metric("Word Accuracy", f"{accuracy:.0f}%", f"{matched}/{total} words")
        with m_col2:
            st.markdown("**Whisper heard:**")
            st.info(transcript if transcript else "(No speech detected)")

        st.markdown("**Word-by-Word Breakdown:**")
        st.markdown(
            f'<div style="font-size:1.15rem;line-height:1.8;padding:0.75rem;border-radius:0.5rem;'
            f'border:1px solid rgba(128,128,128,0.3);">{diff_html}</div>',
            unsafe_allow_html=True,
        )

        if missed_words:
            st.markdown("**Words to Focus On:**")
            for word in missed_words:
                st.markdown(f"- **{word}** — `/{ipa.convert(word)}/`")
        elif total > 0:
            st.success("Clean match on every word!")

        st.divider()

        # 4. Direct A/B Comparison
        st.caption("A/B Comparison")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.caption("Native Speaker")
            st.audio(normal_audio, format="audio/mp3")
        with cmp2:
            st.caption("Your Recording")
            st.audio(wav_bytes, format="audio/wav")