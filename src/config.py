"""Hyperparameters, paths, and constants for DeepCTC."""

import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LJSPEECH_DIR = os.path.join(DATA_DIR, "LJSpeech-1.1")
WAVS_DIR = os.path.join(LJSPEECH_DIR, "wavs")
METADATA_CSV = os.path.join(LJSPEECH_DIR, "metadata.csv")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
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
# lowercase a-z, space, apostrophe
VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz '")

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
