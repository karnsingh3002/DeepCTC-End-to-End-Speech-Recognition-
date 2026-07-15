# DeepCTC

End-to-end speech recognition on LJ Speech: STFT spectrograms → DeepSpeech2-style
CNN + Bidirectional GRU acoustic model → CTC loss/decoding.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9-3.11 (3.12 also works with TensorFlow >= 2.16). On Apple
Silicon, `tensorflow-metal` is installed automatically for GPU acceleration.

`DATA_DIR` and `CHECKPOINT_DIR` default to local paths under the repo (`data/`,
`checkpoints/`) but can be overridden via environment variable — e.g. on Colab
with Drive mounted, set these before running `train.py`:

```python
import os
os.environ["DEEPCTC_DATA_DIR"] = "/content/drive/MyDrive/deepctc_data"
os.environ["DEEPCTC_CHECKPOINT_DIR"] = "/content/drive/MyDrive/deepctc_checkpoints"
```

## Data

Download and extract LJ Speech into `data/LJSpeech-1.1/`:

```bash
make data
```

This fetches the ~2.6GB archive from
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2, extracts 13,100
wav clips + `metadata.csv`, and removes the archive. The train/val split
(90/10, seed=42) is computed on the fly from `metadata.csv` by
`src/dataset.py` — no separate split files are written.

## Sanity checks

```bash
python -m src.vocab    # prints vocab + a text encode/decode round-trip
python -m src.audio    # saves assets/sample_spectrogram.png
python -m src.model    # prints model summary
python -m src.dataset  # builds a small batch and prints shapes
```

## Training

```bash
# Smoke test on a small subset first:
python -m src.train --epochs 5 --subset 200

# Full run:
python -m src.train --epochs 100 --batch_size 32 --lr 1e-4
```

Flags: `--epochs`, `--batch_size`, `--lr`, `--checkpoint_dir`, `--subset`
(train/val on only N examples — useful for smoke tests). Checkpoints
(best `val_loss`) are saved to `checkpoints/deepctc.keras`. Training uses
`ModelCheckpoint`, `EarlyStopping` (patience=8, restores best weights), and
`ReduceLROnPlateau`, so an interrupted run can simply be restarted — it will
overwrite the checkpoint only when validation loss improves. A loss curve is
saved to `assets/loss_curve.png` when training finishes.

Full training to reach strong accuracy takes tens of epochs and multiple
hours on a GPU.

## Evaluation

```bash
python -m src.evaluate --write_results
```

Runs the checkpoint over the full validation set and prints/appends CER,
WER, and accuracy (1 - CER) to `RESULTS.md`.

## Inference

```bash
python -m src.infer --wav path/to/file.wav
```

Prints the greedy-CTC-decoded transcript for a single wav file.

## Project layout

```
src/
  config.py     hyperparameters, paths, constants
  vocab.py      character vocabulary, encode/decode text
  audio.py      STFT spectrogram extraction
  dataset.py    tf.data pipeline (load, pad, batch)
  model.py      DeepSpeech2-style CNN + BiGRU model builder
  losses.py     CTC loss function
  decode.py     greedy CTC decoding
  train.py      training entry point (CLI)
  evaluate.py   CER/WER/accuracy on the val set
  infer.py      transcribe a single wav file (CLI)
```
