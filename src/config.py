from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
FEATURE_COLS_PATH = ARTIFACT_DIR / "feature_cols.pkl"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

TARGET_SR = 22050
CHUNK_DURATION = 5
HOP_DURATION = 5
N_MFCC = 13

AI_THRESHOLD = 0.5
MIN_AUDIO_DURATION = 5

SUPPORTED_AUDIO_TYPES = ["mp3", "wav", "flac", "m4a", "ogg"]