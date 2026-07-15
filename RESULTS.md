# Results

## Status: two pipeline bugs fixed, smoke test re-verified, full training not yet run

Everything below is from a **smoke test** (`python -m src.train --epochs 5 --subset 200
--batch_size 8`), run to confirm the end-to-end pipeline works: data loading, STFT
feature extraction, the CNN+BiGRU model, CTC loss, checkpointing, greedy decoding,
and evaluation all execute without errors and without NaN losses. It was trained on
only 200 of the 13,100 clips for 5 epochs, so the metrics below are **not**
representative of model quality — they only demonstrate correctness of the pipeline.

The reference target (~91.6% character accuracy, ~12.7 validation CTC loss) requires
a full run: `python -m src.train --epochs 100 --batch_size 32 --lr 1e-4` on the full
11,790/1,310 train/val split, which takes tens of epochs and multiple hours on a GPU.
Re-run `python -m src.evaluate --write_results` after that completes to append real
results to this file.

## Bug fixes (2026-07-15)

Two bugs were found and fixed before re-running the smoke test, both of which were
capping validation loss well above the target:

1. **Vocabulary too narrow, corrupting label lengths.** `VOCAB_CHARS` covered only
   `a-z`, space, and apostrophe, so any punctuation in LJSpeech's normalized
   transcripts (periods, commas, quotes, hyphens, etc.) fell into StringLookup's OOV
   bucket — index 0, the same index used for padding. `losses.py` derived
   `label_length` by counting non-zero entries in the padded label, so an embedded
   OOV character (not just trailing padding) silently undercounted the true label
   length, corrupting the CTC target for a large fraction of examples. Fixed by:
   - Expanding `VOCAB_CHARS` to every character actually present in
     `metadata.csv`'s normalized_text column (49 characters — see below).
   - No longer inferring `label_length` from the padded batch at all.
     `dataset.py` now captures the true length via `tf.shape(label)[0]` *before*
     padding and threads it through the pipeline as a third tensor
     (`spectrogram, label, label_length`), so it's exact regardless of vocab
     coverage. `losses.py`'s `ctc_loss` consumes it directly.
2. **Global (whole-clip) spectrogram normalization instead of per-frame.**
   `audio.py` computed a single scalar mean/std over the entire (time, freq)
   spectrogram. Fixed to normalize each time frame independently across its
   frequency bins (`axis=1, keepdims=True`), matching standard DeepSpeech2/Keras
   CTC-ASR practice.

Because the standard `model.compile(loss=fn)` dispatch requires `y_true` and
`y_pred` to share the same nested structure — and rejects a `(label, label_length)`
tuple against a flat prediction tensor — `model.py` now defines `CTCModel`, a
`tf.keras.Model` subclass with a custom `train_step`/`test_step` that calls
`losses.ctc_loss` directly on the raw `(spectrogram, label, label_length)` batch.
`train.py`, `evaluate.py`, and `infer.py` load/save this class directly (registered
via `keras.saving.register_keras_serializable`).

### New VOCAB_CHARS (49 characters, computed from the full corpus)

```
' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', '[', ']',
'a'-'z',
'à', 'â', 'è', 'é', 'ê', 'ü', '’', '“', '”'
```

## Smoke-test run (post-fix)

- Train examples: 200, Val examples: 20 (subset of the full 11,790 / 1,310 split)
- Epochs: 5, batch size: 8, learning rate: 1e-4
- Train CTC loss: 531.58 -> 304.93
- Val CTC loss: 429.24 -> 344.90 (best, restored by EarlyStopping) -> 396.52 (epoch 5)
- No NaN losses at any step; train and val loss both trend downward overall.
- Sample predictions now show the true text with full punctuation preserved, e.g.
  `"at that station the safes were given out, heavy with shot, not gold; the
  thieves went on to dover, and by-and-by,"` — confirming punctuation is no longer
  silently dropped from labels.

![Loss curve](assets/loss_curve.png)

## Evaluation (smoke-test checkpoint, 20 val examples)

- CER: 0.8837
- WER: 0.9970
- Accuracy (1 - CER): 0.1163

These numbers reflect a model trained on 200 examples for 5 epochs, not the full
dataset — still expected to be low. They confirm `src/evaluate.py`'s CER/WER/
accuracy computation runs correctly end-to-end against the fixed pipeline.

## Sanity checks performed

- Vocab encode/decode round-trips exactly for in-vocab text (`python -m src.vocab`);
  genuinely out-of-corpus characters (digits, £, em-dash) still correctly fall back
  to OOV.
- Sample spectrogram (`assets/sample_spectrogram.png`) shows clear formant bands and
  silence gaps, consistent with real speech rather than noise.
- Model builds with the expected shapes: freq dim 193 -> 97 -> 49 after the two
  stride-2 conv layers; ~8.08M parameters (output layer now 51-wide: 49 vocab chars
  + OOV/pad + CTC blank).
- `src/dataset.py` train/val split: 11,790 train / 1,310 val (90/10, seed=42) from
  the full 13,100-clip metadata.csv; batches now yield
  `(spectrogram, label, label_length)`.
- `python -m src.infer --wav <file>` runs end-to-end and prints a transcript
  (low-quality on the smoke-test checkpoint, as expected for 5 epochs on 200
  examples).
