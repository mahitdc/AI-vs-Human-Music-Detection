import json

import joblib
import librosa
import numpy as np

from config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLS_PATH,
    METADATA_PATH,
    TARGET_SR,
    CHUNK_DURATION,
    HOP_DURATION,
    AI_THRESHOLD,
    MIN_AUDIO_DURATION,
)

from features import extract_features_from_audio


def load_artifacts():
    """Load trained model, scaler, feature column order, and metadata."""

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    metadata = None
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, scaler, feature_cols, metadata


def get_score_band(probability):
    """
    Convert the winning class probability into a readable score band.
    This is not a calibrated confidence interval.
    """

    percent = probability * 100

    if percent >= 90:
        return "Very High"
    if percent >= 75:
        return "High"
    if percent >= 60:
        return "Moderate"

    return "Low"


def features_to_vector(feature_dict, feature_cols, scaler):
    """Convert feature dictionary into scaled model input."""

    vector = np.array([[feature_dict[col] for col in feature_cols]])
    return scaler.transform(vector)


def predict_audio_file(file_path):
    """
    Run the deployed inference pipeline on one audio file.

    Pipeline:
    1. Load audio.
    2. Split into non-overlapping 5-second chunks.
    3. Extract 104 DSP features per chunk.
    4. Predict P(AI) for each chunk.
    5. Average chunk probabilities for the final song-level prediction.
    """

    model, scaler, feature_cols, metadata = load_artifacts()

    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not load audio file: {exc}") from exc

    duration = len(y) / sr

    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio is too short. Minimum duration is {MIN_AUDIO_DURATION} seconds."
        )

    chunk_samples = int(CHUNK_DURATION * sr)
    hop_samples = int(HOP_DURATION * sr)

    starts = list(range(0, len(y) - chunk_samples + 1, hop_samples))

    if not starts:
        raise ValueError("No valid audio chunks could be created.")

    chunk_results = []
    chunk_probabilities = []

    for chunk_index, start in enumerate(starts, start=1):
        end = start + chunk_samples
        chunk = y[start:end]

        feature_dict = extract_features_from_audio(chunk, sr)
        x = features_to_vector(feature_dict, feature_cols, scaler)

        probabilities = model.predict_proba(x)[0]
        human_probability = float(probabilities[0])
        ai_probability = float(probabilities[1])

        chunk_prediction = "AI" if ai_probability >= AI_THRESHOLD else "Human"

        chunk_probabilities.append(ai_probability)

        chunk_results.append(
            {
                "chunk": chunk_index,
                "start_time": start / sr,
                "end_time": end / sr,
                "ai_probability": ai_probability,
                "human_probability": human_probability,
                "prediction": chunk_prediction,
            }
        )

    avg_ai_probability = float(np.mean(chunk_probabilities))
    avg_human_probability = 1.0 - avg_ai_probability

    final_prediction = (
        "AI" if avg_ai_probability >= AI_THRESHOLD else "Human"
    )

    winning_probability = max(avg_ai_probability, avg_human_probability)
    score_band = get_score_band(winning_probability)

    return {
        "prediction": final_prediction,
        "ai_probability": avg_ai_probability,
        "human_probability": avg_human_probability,
        "score_band": score_band,
        "duration": duration,
        "num_chunks": len(chunk_results),
        "chunk_results": chunk_results,
        "metadata": metadata,
    }