import librosa
import numpy as np

def extract_pitch(wav, sr):
    pitches, magnitudes = librosa.core.piptrack(y=wav, sr=sr)
    return pitches

def calculate_mean_pitch(pitches):
    return np.mean([np.mean(p[p > 0]) for p in pitches.T if np.sum(p > 0) > 0])

def process_audio(audio, sr):
    pitches = extract_pitch(audio, sr = sr)
    mean_pitch = calculate_mean_pitch(pitches)
    energy = librosa.feature.rms(y=audio)
    mean_energy = np.mean(energy[~np.isnan(energy)])

    return mean_pitch, mean_energy