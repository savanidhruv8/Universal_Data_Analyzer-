import os
import tempfile
import warnings
import numpy as np
import streamlit as st
import soundfile as sf
import scipy.signal
from pydub import AudioSegment

warnings.filterwarnings("ignore")

# ----------------------------
# Load ANY audio to numpy (using pydub only)
# ----------------------------
def load_audio_to_numpy(file_path: str):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Mono
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 1:
        samples = samples.astype(np.float32) / 128.0
    elif audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / (2 ** (8 * audio.sample_width - 1))
    return samples, audio.frame_rate


# ----------------------------
# Advanced Audio Preprocessor (NO LIBROSA!)
# ----------------------------
class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio
        gcd = np.gcd(int(orig_sr), int(self.target_sr))
        up = self.target_sr // gcd
        down = orig_sr // gcd
        max_ratio = 10
        if up > max_ratio or down > max_ratio:
            current_sr = orig_sr
            current_audio = audio
            while current_sr != self.target_sr:
                next_sr = current_sr
                if current_sr < self.target_sr:
                    next_sr = min(current_sr * 2, self.target_sr)
                else:
                    next_sr = max(current_sr // 2, self.target_sr)
                gcd_step = np.gcd(int(current_sr), int(next_sr))
                up_step = next_sr // gcd_step
                down_step = current_sr // gcd_step
                current_audio = scipy.signal.resample_poly(current_audio, up_step, down_step, axis=-1)
                current_sr = next_sr
            return current_audio
        else:
            return scipy.signal.resample_poly(audio, up, down, axis=-1)

    def normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        scalar = 10 ** (target_db / 20) / peak
        audio = audio * scalar
        return np.clip(audio, -1.0, 1.0)

    def highpass_filter(self, audio: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        nyquist = self.target_sr / 2
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='high')
        return scipy.signal.filtfilt(b, a, audio)

    def spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        n_fft = 2048
        hop_length = 512
        window = scipy.signal.get_window('hann', n_fft)
        f, t, Zxx = scipy.signal.stft(
            audio, fs=self.target_sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        noise_frames = min(10, mag.shape[1] // 4)
        noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True) if noise_frames > 0 else np.mean(mag, axis=1, keepdims=True)
        cleaned_mag = np.maximum(mag - 2.0 * noise_est, 0.1 * mag)
        cleaned_Zxx = cleaned_mag * np.exp(1j * phase)
        _, enhanced = scipy.signal.istft(
            cleaned_Zxx, fs=self.target_sr, window=window,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )
        return enhanced

    def remove_all_silence(self, audio: np.ndarray, top_db: int = 30, min_silence_len: float = 0.3) -> np.ndarray:
        """
        Remove ALL silent segments (including internal gaps).
        Keeps only non-silent chunks and concatenates them.
        """
        frame_length = int(self.target_sr * 0.05)  # 50ms frames
        hop_length = frame_length // 2
        min_silence_samples = int(min_silence_len * self.target_sr)

        # Compute energy per frame
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energies.append(np.sum(frame ** 2))
        if not energies:
            return audio

        energies = np.array(energies)
        rms_energies = np.sqrt(energies / frame_length)
        threshold = np.max(rms_energies) / (10 ** (top_db / 20))

        # Find non-silent regions
        non_silent = rms_energies > threshold
        segments = []
        start = None
        for i, is_active in enumerate(non_silent):
            if is_active and start is None:
                start = i * hop_length
            elif not is_active and start is not None:
                end = i * hop_length + frame_length
                segments.append((start, end))
                start = None
        if start is not None:
            segments.append((start, len(audio)))

        # Concatenate non-silent segments
        if not segments:
            return np.array([])
        cleaned = np.concatenate([audio[s:e] for s, e in segments])
        return cleaned

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = self.resample(audio, sr)
        audio = self.spectral_subtraction(audio)      # Reduce background noise
        audio = self.remove_all_silence(audio)       # Remove ALL silent gaps
        audio = self.highpass_filter(audio)          # Remove low-frequency rumble
        audio = self.normalize(audio, target_db=-3.0)
        return audio


# ----------------------------
# Model Recommendation (Post-Cleaning)
# ----------------------------
def suggest_models(duration: float):
    if duration < 0.5:
        return []
    elif duration <= 30:
        return [
            "Whisper (base/small) – for transcription",
            "Wav2Vec2 (facebook/wav2vec2-base-960h) – English ASR",
            "PANNs – for sound classification"
        ]
    else:
        return [
            "Whisper (medium/large-v3) with chunking",
            "NVIDIA NeMo ASR – for long-form audio",
            "Custom pipeline with sliding window"
        ]


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Audio Cleaner", layout="centered")

# Add back button with improved styling
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: #1a365d;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s;
        height: auto;
        min-height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: auto;
        min-width: 80px;
        box-sizing: border-box;
        white-space: nowrap;
    }
    div.stButton > button:first-child:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("⬅️ Back", key="audio_back_button"):
        st.switch_page("main.py")

st.title("🎙️ Clean & Standardize Audio")
st.markdown("""
Upload any audio file (WAV, MP3, M4A, MP4). We'll:
- Convert to **16-bit mono WAV @ 16kHz**
- **Remove background noise**
- **Remove ALL silent gaps** (including between speakers)
- **Normalize to -3 dB peak**
- **Reject if final length < 0.5s**
""")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "mp4", "ogg", "flac"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            orig_audio, orig_sr = load_audio_to_numpy(input_path)
        except Exception as e:
            st.error(f"❌ Failed to load audio: {e}")
            st.stop()

        orig_duration = len(orig_audio) / orig_sr
        st.audio(input_path, format="audio/wav")
        st.info(f"Original: {orig_duration:.2f}s @ {orig_sr} Hz")

        # Preprocess
        preprocessor = AudioPreprocessor(target_sr=16000)
        try:
            processed_audio = preprocessor.preprocess_audio(orig_audio, orig_sr)
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            st.stop()

        final_duration = len(processed_audio) / preprocessor.target_sr

        if final_duration < 0.5:
            st.error(f"❌ Final audio too short ({final_duration:.2f}s). Minimum: 0.5s")
            st.stop()

        # Save as 16-bit WAV
        output_path = os.path.join(tmp_dir, "cleaned_audio.wav")
        sf.write(output_path, processed_audio.astype(np.float32), preprocessor.target_sr, subtype='PCM_16')

        st.success("✅ Audio cleaned and standardized!")
        st.subheader("🔊 Cleaned Audio")
        st.audio(output_path, format="audio/wav")
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Cleaned Audio (16kHz, mono, 16-bit)",
                f,
                file_name="cleaned_audio.wav"
            )

        # Model suggestions AFTER cleaning
        st.subheader("🤖 Recommended Models")
        models = suggest_models(final_duration)
        st.write(f"**Final audio**: {final_duration:.2f}s @ 16kHz")
        for model in models:
            st.markdown(f"- {model}")

else:
    st.info("👆 Upload an audio file to get started!")