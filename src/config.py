"""Hyperparameters, paths, and constants for DeepCTC."""

import os

# --- Paths ---
# DATA_DIR and CHECKPOINT_DIR default to local paths under the repo, but can be
# overridden via environment variable — e.g. on Colab with Drive mounted:
#   os.environ["DEEPCTC_DATA_DIR"] = "/content/drive/MyDrive/deepctc_data"
#   os.environ["DEEPCTC_CHECKPOINT_DIR"] = "/content/drive/MyDrive/deepctc_checkpoints"
# set before importing src.config/src.train, so nothing else in the codebase changes.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DEEPCTC_DATA_DIR", os.path.join(ROOT_DIR, "data"))
LJSPEECH_DIR = os.path.join(DATA_DIR, "LJSpeech-1.1")
WAVS_DIR = os.path.join(LJSPEECH_DIR, "wavs")
METADATA_CSV = os.path.join(LJSPEECH_DIR, "metadata.csv")
CHECKPOINT_DIR = os.environ.get("DEEPCTC_CHECKPOINT_DIR", os.path.join(ROOT_DIR, "checkpoints"))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
RESULTS_MD = os.path.join(ROOT_DIR, "RESULTS.md")

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

# --- Dataset split ---
VAL_SPLIT = 0.1
SPLIT_SEED = 42

# --- Audio / STFT ---
SAMPLE_RATE = 22050
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384
SPEC_FEATURE_DIM = FFT_LENGTH // 2 + 1  # 193
POWER_COMPRESSION = 0.5
NORM_EPS = 1e-10

# --- Vocabulary ---
# Every distinct lowercased character present in metadata.csv's normalized_text
# column (3rd pipe-delimited field) — a-z plus the punctuation and accented
# characters LJSpeech's normalized transcripts actually contain. Recompute with:
#   python -c "import pandas as pd; df = pd.read_csv('data/LJSpeech-1.1/metadata.csv', sep='|', header=None, names=['file_id','raw_text','normalized_text'], quoting=3, keep_default_na=False); chars = set(); [chars.update(t.lower()) for t in df['normalized_text']]; chars.discard(' '); print([' '] + sorted(chars))"
VOCAB_CHARS = [
    " ", "!", '"', "'", "(", ")", ",", "-", ".", ":", ";", "?", "[", "]",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "à", "â", "è", "é", "ê", "ü", "’", "“", "”",
]

# --- Model ---
RNN_UNITS = 256
RNN_LAYERS = 5
DROPOUT = 0.3

# --- Training defaults ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5

# CTC time-downsampling factor introduced by the model's strided conv layers
# (2 conv layers each stride 2 in time => divide by 4)
TIME_DOWNSAMPLE_FACTOR = 4
