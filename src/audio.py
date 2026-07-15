"""STFT spectrogram extraction for raw wav files."""

import tensorflow as tf

from src.config import (
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    NORM_EPS,
    POWER_COMPRESSION,
)


def load_wav(path):
    """Path (tf.string scalar) -> 1D float32 waveform tensor in [-1, 1]."""
    audio_bytes = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    return tf.squeeze(audio, axis=-1)


def wav_to_spectrogram(waveform):
    """1D float32 waveform -> (time, freq) normalized magnitude spectrogram."""
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.pow(spectrogram, POWER_COMPRESSION)

    mean = tf.math.reduce_mean(spectrogram, axis=1, keepdims=True)
    std = tf.math.reduce_std(spectrogram, axis=1, keepdims=True)
    spectrogram = (spectrogram - mean) / (std + NORM_EPS)
    return spectrogram


def path_to_spectrogram(path):
    """Path (tf.string scalar) -> normalized (time, freq) spectrogram."""
    return wav_to_spectrogram(load_wav(path))


if __name__ == "__main__":
    import glob
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import ASSETS_DIR, WAVS_DIR

    wav_files = sorted(glob.glob(os.path.join(WAVS_DIR, "*.wav")))
    if not wav_files:
        raise SystemExit(
            f"No wav files found in {WAVS_DIR}. Download the dataset first."
        )

    sample_path = wav_files[0]
    spec = path_to_spectrogram(tf.constant(sample_path))
    print(f"Sample file: {sample_path}")
    print(f"Spectrogram shape (time, freq): {spec.shape}")
    print(f"Mean: {tf.reduce_mean(spec):.4f}  Std: {tf.math.reduce_std(spec):.4f}")

    os.makedirs(ASSETS_DIR, exist_ok=True)
    out_path = os.path.join(ASSETS_DIR, "sample_spectrogram.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.transpose(spec).numpy(), origin="lower", aspect="auto", cmap="magma")
    plt.title(f"Spectrogram: {os.path.basename(sample_path)}")
    plt.xlabel("Time frame")
    plt.ylabel("Frequency bin")
    plt.colorbar(label="Normalized magnitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sample spectrogram plot to {out_path}")
