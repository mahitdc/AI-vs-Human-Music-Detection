import numpy as np
import librosa
from scipy.stats import skew, kurtosis

from config import N_MFCC


def extract_features_from_audio(y, sr):
    
    """
    Extract the 104 DSP features used by the trained model.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    dict
        Feature dictionary matching the training feature columns.
    """

    rms = librosa.feature.rms(y=y).flatten()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
    zcr = librosa.feature.zero_crossing_rate(y).flatten()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    flatness = librosa.feature.spectral_flatness(y=y).flatten()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)

    features = {
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
        "rms_skew": skew(rms),
        "rms_kurtosis": kurtosis(rms),

        "centroid_mean": np.mean(centroid),
        "centroid_var": np.var(centroid),
        "centroid_skew": skew(centroid),
        "centroid_kurtosis": kurtosis(centroid),

        "bandwidth_mean": np.mean(bandwidth),
        "bandwidth_var": np.var(bandwidth),

        "rolloff_mean": np.mean(rolloff),
        "rolloff_var": np.var(rolloff),

        "zcr_mean": np.mean(zcr),
        "zcr_var": np.var(zcr),

        "tempo": float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo),
        "num_beats": len(beats),

        "dynamic_range": float(np.max(rms) - np.min(rms)),
        "rms_variability": float(np.std(rms) / (np.mean(rms) + 1e-9)),

        "flatness_mean": np.mean(flatness),
        "flatness_var": np.var(flatness),
        "flatness_skew": skew(flatness),
        "flatness_kurtosis": kurtosis(flatness),

        "onset_mean": np.mean(onset_env),
        "onset_var": np.var(onset_env),
        "onset_skew": skew(onset_env),
        "onset_kurtosis": kurtosis(onset_env),

        "tonnetz_mean": np.mean(tonnetz),
        "tonnetz_var": np.var(tonnetz),
    }

    for i in range(12):
        features[f"chroma_{i + 1}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i + 1}_var"] = np.var(chroma[i])

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i + 1}_var"] = np.var(mfcc[i])
        features[f"mfcc_delta_{i + 1}_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_delta_{i + 1}_var"] = np.var(mfcc_delta[i])

    return features
