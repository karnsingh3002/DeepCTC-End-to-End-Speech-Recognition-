"""tf.data pipeline: LJ Speech metadata -> batched (spectrogram, label, label_length) triples."""

import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.audio import path_to_spectrogram
from src.config import (
    BATCH_SIZE,
    METADATA_CSV,
    SPLIT_SEED,
    VAL_SPLIT,
    WAVS_DIR,
)
from src.model import compute_time_steps_after_conv_tf
from src.vocab import encode_text

AUTOTUNE = tf.data.AUTOTUNE


def load_metadata():
    """Read metadata.csv (pipe-delimited: file_id|raw_text|normalized_text)."""
    df = pd.read_csv(
        METADATA_CSV,
        sep="|",
        header=None,
        names=["file_id", "raw_text", "normalized_text"],
        quoting=3,  # csv.QUOTE_NONE — LJSpeech text may contain stray quotes
        keep_default_na=False,
    )
    df["wav_path"] = df["file_id"].apply(lambda fid: os.path.join(WAVS_DIR, f"{fid}.wav"))
    return df


def train_val_split(df=None, val_split=VAL_SPLIT, seed=SPLIT_SEED):
    if df is None:
        df = load_metadata()
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _load_example(wav_path, text):
    spectrogram = path_to_spectrogram(wav_path)
    label = encode_text(text)
    # Captured before padding, so it's exact regardless of vocab/OOV coverage —
    # unlike counting non-zero entries after padded_batch, which breaks if any
    # in-sequence character happens to map to the same index (0) as padding.
    label_length = tf.shape(label)[0]
    return spectrogram, label, label_length


def _is_ctc_feasible_tf(spectrogram, label, label_length):
    time_steps = tf.shape(spectrogram)[0]
    input_length = compute_time_steps_after_conv_tf(time_steps)
    return input_length >= label_length


def build_dataset(df, batch_size=BATCH_SIZE, shuffle=True, report_drops=True):
    """Build a padded, batched, prefetched tf.data.Dataset from a metadata dataframe."""
    wav_paths = tf.constant(df["wav_path"].values)
    texts = tf.constant(df["normalized_text"].values)

    ds = tf.data.Dataset.from_tensor_slices((wav_paths, texts))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SPLIT_SEED)

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE)

    if report_drops:
        total, kept = ds.reduce(
            (0, 0),
            lambda acc, ex: (acc[0] + 1, acc[1] + tf.cast(_is_ctc_feasible_tf(*ex), tf.int32)),
        )
        total, kept = total.numpy(), kept.numpy()
        dropped = total - kept
        if dropped:
            print(
                f"[dataset] dropped {dropped}/{total} examples where "
                f"label_length > CTC input_length"
            )

    ds = ds.filter(_is_ctc_feasible_tf)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([None, None], [None], []),
        padding_values=(0.0, tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int32)),
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds


if __name__ == "__main__":
    train_df, val_df = train_val_split()
    print(f"Train examples: {len(train_df)}  Val examples: {len(val_df)}")

    train_ds = build_dataset(train_df.head(64), batch_size=8)
    for specs, labels, label_lengths in train_ds.take(1):
        print(f"Batch spectrogram shape: {specs.shape}")
        print(f"Batch label shape: {labels.shape}")
        print(f"Batch label_length shape/values: {label_lengths.shape} {label_lengths.numpy()}")
