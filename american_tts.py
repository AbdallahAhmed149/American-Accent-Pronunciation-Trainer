import asyncio
import difflib
import os
import re
import tempfile
import wave
import edge_tts
import eng_to_ipa as ipa
import numpy as np
import pygame
import sounddevice as sd
from faster_whisper import WhisperModel

VOICES = {
    "1": ("Andrew (Male - Conversational)", "en-US-AndrewMultilingualNeural"),
    "2": ("Ava (Female - Conversational)", "en-US-AvaMultilingualNeural"),
    "3": ("Brian (Male - Clear/Warm)", "en-US-BrianMultilingualNeural"),
    "4": ("Emma (Female - Clear/Warm)", "en-US-EmmaMultilingualNeural"),
}

SAMPLE_RATE = 16000  # 16 kHz is Whisper's native sample rate
NATIVE_AUDIO = os.path.join(tempfile.gettempdir(), "american_tts_native.mp3")
USER_AUDIO = os.path.join(tempfile.gettempdir(), "american_tts_user.wav")

# ANSI colors for terminal highlighting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def normalize_words(text: str) -> list[str]:
    """Lowercase and strip punctuation while keeping contractions like don't."""
    return re.findall(r"[a-z']+", text.lower())


def evaluate_pronunciation(target: str, transcript: str):
    """Compare target sentence against Whisper's transcript and highlight mismatches."""
    target_words = normalize_words(target)
    spoken_words = normalize_words(transcript)

    if not target_words:
        return

    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
    highlighted = []
    missed_words = []
    matched_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in target_words[i1:i2]:
                highlighted.append(f"{GREEN}{w}{RESET}")
                matched_count += 1
        elif tag == "replace":
            expected = " ".join(target_words[i1:i2])
            heard = " ".join(spoken_words[j1:j2])
            highlighted.append(f"{RED}{expected}{RESET} {YELLOW}(heard: {heard}){RESET}")
            missed_words.extend(target_words[i1:i2])
        elif tag == "delete":
            for w in target_words[i1:i2]:
                highlighted.append(f"{RED}[{w}]{RESET}")
                missed_words.append(w)
        elif tag == "insert":
            extra = " ".join(spoken_words[j1:j2])
            highlighted.append(f"{YELLOW}(+{extra}){RESET}")

    accuracy = (matched_count / len(target_words)) * 100

    print(f"\n  Heard:    \"{transcript}\"")
    print(f"  Result:   {' '.join(highlighted)}")
    print(f"  Accuracy: {accuracy:.0f}% ({matched_count}/{len(target_words)} words)")

    if missed_words:
        unique_missed = list(dict.fromkeys(missed_words))
        print("  Focus on:")
        for word in unique_missed:
            print(f"    - {word}: /{ipa.convert(word)}/")


def transcribe_audio(model: WhisperModel, path: str) -> str:
    """Run local Whisper transcription on the recorded WAV file."""
    segments, _ = model.transcribe(path, language="en", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


async def play_file(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
    pygame.mixer.music.unload()


async def synthesize(text: str, voice: str, rate: str = "+0%"):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(NATIVE_AUDIO)


def record_user_voice() -> bool:
    input("  Press [Enter] to start recording...")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input("  Recording... Press [Enter] to stop.")

    if not frames:
        return False

    audio_data = np.concatenate(frames, axis=0)
    with wave.open(USER_AUDIO, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    return True


async def compare_playback():
    print("  [Compare] Native speaker...")
    await play_file(NATIVE_AUDIO)
    await asyncio.sleep(0.5)
    print("  [Compare] Your voice...")
    await play_file(USER_AUDIO)


async def main():
    print("Loading Whisper (base.en) model...")
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

    pygame.mixer.init()
    voice = VOICES["1"][1]
    shadow_mode = True
    record_mode = True
    last_sentence = ""
    has_recording = False

    print("\n=== American Accent & Pronunciation Trainer ===")
    print("Commands: /shadow, /rec, /compare, /voice, /r (repeat), /q (quit)\n")

    while True:
        text = input("Enter sentence: ").strip()
        if not text:
            continue

        if text == "/q":
            break
        elif text == "/shadow":
            shadow_mode = not shadow_mode
            print(f"Shadowing mode: {'ON' if shadow_mode else 'OFF'}\n")
            continue
        elif text == "/rec":
            record_mode = not record_mode
            print(f"Recording mode: {'ON' if record_mode else 'OFF'}\n")
            continue
        elif text == "/compare":
            if has_recording and os.path.exists(NATIVE_AUDIO):
                await compare_playback()
                print()
            else:
                print("Record a sentence first.\n")
            continue
        elif text == "/voice":
            for k, (label, _) in VOICES.items():
                print(f"  {k}. {label}")
            choice = input("Pick a voice: ").strip()
            if choice in VOICES:
                voice = VOICES[choice][1]
                print(f"Switched to {VOICES[choice][0]}.\n")
            continue
        elif text == "/r":
            text = last_sentence
            if not text:
                print("Nothing to repeat yet.\n")
                continue

        last_sentence = text
        print(f"IPA: /{ipa.convert(text)}/")

        if shadow_mode:
            print("  [1/2] Slow (-25%)...")
            await synthesize(text, voice, rate="-25%")
            await play_file(NATIVE_AUDIO)
            await asyncio.sleep(1.5)
            print("  [2/2] Normal (1.0x)...")
            await synthesize(text, voice, rate="+0%")
            await play_file(NATIVE_AUDIO)
        else:
            await synthesize(text, voice, rate="+0%")
            await play_file(NATIVE_AUDIO)

        if record_mode:
            if record_user_voice():
                has_recording = True
                print("  Analyzing pronunciation...")
                transcript = await asyncio.to_thread(transcribe_audio, whisper_model, USER_AUDIO)
                evaluate_pronunciation(text, transcript)
                print()
                await compare_playback()

        print()

    pygame.mixer.quit()


if __name__ == "__main__":
    asyncio.run(main())