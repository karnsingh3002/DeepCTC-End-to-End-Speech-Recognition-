"""Transcribe a single wav file with a trained checkpoint.

Usage:
    python -m src.infer --wav path/to/file.wav
"""

import argparse
import os

import tensorflow as tf

from src.audio import path_to_spectrogram
from src.config import CHECKPOINT_DIR
from src.decode import greedy_decode
from src.model import CTCModel


def transcribe(model, wav_path):
    spectrogram = path_to_spectrogram(tf.constant(wav_path))
    batch = tf.expand_dims(spectrogram, axis=0)
    preds = model.predict(batch, verbose=0)
    return greedy_decode(preds)[0]


def main():
    parser = argparse.ArgumentParser(description="Transcribe a single wav file.")
    parser.add_argument("--wav", type=str, required=True, help="Path to a wav file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "deepctc.keras"),
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.checkpoint, custom_objects={"CTCModel": CTCModel})
    transcript = transcribe(model, args.wav)
    print(transcript)


if __name__ == "__main__":
    main()
