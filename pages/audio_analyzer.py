import streamlit as st
import soundfile as sf
import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from pydub import AudioSegment
import os
import uuid
import io
try:
    from speechbrain.pretrained import VAD
except ImportError:
    st.error("Failed to import speechbrain VAD. Ensure 'speechbrain' is installed correctly with 'pip install speechbrain'.")
    st.stop()

# ------------------ Audio Processing Functions ------------------

def convert_to_wav(input_file, output_path):
    """Convert audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1)  # Mono channel
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise Exception(f"Error converting to WAV: {e}")

def apply_wave_format(audio_path):
    """Ensure audio is in 16-bit PCM WAV format."""
    try:
        data, sr = sf.read(audio_path)
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
            sf.write(audio_path, data, sr, subtype='PCM_16')
        return audio_path
    except Exception as e:
        raise Exception(f"Error applying wave format: {e}")

def resample_audio(audio_path, target_sr=16000):
    """Resample audio to target sample rate."""
    try:
        data, sr = librosa.load(audio_path, sr=None)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sf.write(audio_path, data, target_sr)
        return audio_path, target_sr
    except Exception as e:
        raise Exception(f"Error resampling audio: {e}")

def normalize_audio(audio_path):
    """Normalize audio to prevent clipping."""
    try:
        data, sr = sf.read(audio_path)
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * 0.9
            sf.write(audio_path, data, sr)
        return audio_path
    except Exception as e:
        raise Exception(f"Error normalizing audio: {e}")

def apply_high_pass_filter(audio_path, cutoff=100):
    """Apply high-pass filter to remove low-frequency noise."""
    try:
        data, sr = sf.read(audio_path)
        sos = signal.butter(10, cutoff, 'highpass', fs=sr, output='sos')
        filtered_data = signal.sosfilt(sos, data)
        sf.write(audio_path, filtered_data, sr)
        return audio_path
    except Exception as e:
        raise Exception(f"Error applying high-pass filter: {e}")

def spectral_subtraction(audio_path, noise_duration=0.5):
    """Apply spectral subtraction for noise reduction."""
    try:
        data, sr = sf.read(audio_path)
        noise_samples = int(noise_duration * sr)
        noise = data[:noise_samples]
        noise_spectrum = np.abs(fft(noise))
        signal_spectrum = fft(data)
        magnitude = np.abs(signal_spectrum)
        phase = np.angle(signal_spectrum)
        clean_magnitude = np.maximum(magnitude - np.mean(noise_spectrum), 0)
        clean_signal = ifft(clean_magnitude * np.exp(1j * phase)).real
        sf.write(audio_path, clean_signal, sr)
        return audio_path
    except Exception as e:
        raise Exception(f"Error applying spectral subtraction: {e}")

def apply_vad(audio_path, output_dir):
    """Apply Voice Activity Detection to remove non-speech segments."""
    try:
        # Initialize VAD model
        vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=os.path.join(output_dir, "tmp_vad"))
        data, sr = sf.read(audio_path)
        temp_path = os.path.join(output_dir, f"temp_{uuid.uuid4()}.wav")
        sf.write(temp_path, data, sr)
        
        # Get speech segments
        boundaries = vad.get_speech_segments(temp_path)
        speech_data = [data[int(start*sr):int(end*sr)] for start, end in boundaries]
        
        if speech_data:
            speech_data = np.concatenate(speech_data)
            sf.write(audio_path, speech_data, sr)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return audio_path
    except Exception as e:
        raise Exception(f"Error applying VAD: {e}")

# ------------------ Algorithm Suggestion ------------------

def suggest_algorithm(main_type, sub_type, duration, sample_rate):
    """Suggest ML model based on task and audio properties."""
    if main_type == "Classification":
        if sub_type == "Emotion Recognition":
            return "Wav2Vec2 Emotion Classifier" if duration > 10 else "CNN-RNN Real-time Emotion Detector"
        elif sub_type == "Speaker Identification":
            return "ECAPA-TDNN (SpeechBrain)"
    elif main_type == "Detection":
        if sub_type == "Keyword Spotting":
            return "YAMNet / ResNet-KWS"
        elif sub_type == "VAD":
            return "SpeechBrain CRDNN VAD Model"
    elif main_type == "Enhancement":
        if sub_type == "Noise Reduction":
            return "DeepFilterNet or RNNoise"
        elif sub_type == "Dereverberation":
            return "WPE from Pyroomacoustics"
    return "No matching model found."

# ------------------ Streamlit UI ------------------

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Audio Enhancer + ML Recommender", layout="centered")
    st.title("🔊 Audio Enhancer + 🧠 ML Model Suggestion")
    
    uploaded_file = st.file_uploader("📁 Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1]
        raw_path = os.path.join("input_" + str(uuid.uuid4()) + "." + file_ext)

        # Save uploaded file
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.read())

        st.subheader("▶️ Original Audio")
        st.audio(raw_path)

        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"enhanced_{uuid.uuid4()}.wav")

        try:
            # Convert non-WAV to WAV
            if file_ext.lower() in ['mp3', 'm4a']:
                input_wav = convert_to_wav(raw_path, output_path)
            else:
                input_wav = output_path
                sf.write(input_wav, *sf.read(raw_path))

            # Apply audio enhancement pipeline
            input_wav = apply_wave_format(input_wav)
            input_wav, sr = resample_audio(input_wav)
            input_wav = normalize_audio(input_wav)
            input_wav = apply_high_pass_filter(input_wav)
            input_wav = spectral_subtraction(input_wav)
            input_wav = apply_vad(input_wav, output_dir)

            st.subheader("✅ Enhanced Audio")
            st.audio(input_wav)

            # ML model suggestion
            st.subheader("🧠 ML Model Suggestion")
            main_type = st.selectbox("Select Main Type", ["Classification", "Detection", "Enhancement"])
            sub_types = {
                "Classification": ["Emotion Recognition", "Speaker Identification"],
                "Detection": ["Keyword Spotting", "VAD"],
                "Enhancement": ["Noise Reduction", "Dereverberation"]
            }
            sub_type = st.selectbox("Select Sub-Type", sub_types[main_type])

            audio_data, sr = sf.read(input_wav)
            duration = len(audio_data) / sr
            suggestion = suggest_algorithm(main_type, sub_type, duration, sr)
            st.success(f"Suggested Model: {suggestion}")

            # Provide download option
            with open(input_wav, "rb") as f:
                st.download_button(
                    "⬇️ Download Enhanced Audio",
                    data=f,
                    file_name="enhanced_audio.wav",
                    mime="audio/wav"
                )

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")

        # Clean up temporary files
        for path in [raw_path, input_wav]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    st.warning(f"⚠️ Could not delete temporary file: {path} ({e})")

if __name__ == "__main__":
    main()