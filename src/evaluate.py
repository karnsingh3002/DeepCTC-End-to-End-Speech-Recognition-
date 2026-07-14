"""Evaluate a trained checkpoint on the validation set: CER, WER, accuracy."""

import argparse
import os

import jiwer
import tensorflow as tf
from tqdm import tqdm

from src.config import BATCH_SIZE, CHECKPOINT_DIR, RESULTS_MD
from src.dataset import build_dataset, train_val_split
from src.decode import greedy_decode
from src.losses import ctc_loss
from src.vocab import decode_indices


def evaluate(checkpoint_path, batch_size=BATCH_SIZE, subset=None):
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={"ctc_loss": ctc_loss})

    _, val_df = train_val_split()
    if subset:
        val_df = val_df.head(subset).reset_index(drop=True)
    val_ds = build_dataset(val_df, batch_size=batch_size, shuffle=False)

    references, hypotheses = [], []
    for specs, labels in tqdm(val_ds, desc="Evaluating"):
        preds = model.predict(specs, verbose=0)
        hypotheses.extend(greedy_decode(preds))
        references.extend(decode_indices(labels))

    # jiwer chokes on empty strings; keep pairs where the reference is non-empty.
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    references, hypotheses = zip(*pairs)

    cer = jiwer.cer(list(references), list(hypotheses))
    wer = jiwer.wer(list(references), list(hypotheses))
    accuracy = 1.0 - cer

    return {
        "cer": cer,
        "wer": wer,
        "accuracy": accuracy,
        "num_examples": len(references),
        "sample_predictions": list(zip(references[:5], hypotheses[:5])),
    }


def write_results_md(metrics, val_loss=None):
    lines = [
        "## Evaluation results\n",
        f"- Validation examples: {metrics['num_examples']}\n",
        f"- CER: {metrics['cer']:.4f}\n",
        f"- WER: {metrics['wer']:.4f}\n",
        f"- Accuracy (1 - CER): {metrics['accuracy']:.4f}\n",
    ]
    if val_loss is not None:
        lines.append(f"- Validation CTC loss: {val_loss:.4f}\n")
    lines.append("\n### Sample predictions\n")
    for ref, hyp in metrics["sample_predictions"]:
        lines.append(f"- true: `{ref}`\n  pred: `{hyp}`\n")

    with open(RESULTS_MD, "a") as f:
        f.writelines(lines)
    print(f"Appended evaluation results to {RESULTS_MD}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepCTC on the validation set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "deepctc.keras"),
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--write_results", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, batch_size=args.batch_size, subset=args.subset)

    print(f"Examples evaluated: {metrics['num_examples']}")
    print(f"CER:      {metrics['cer']:.4f}")
    print(f"WER:      {metrics['wer']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    if args.write_results:
        write_results_md(metrics)


if __name__ == "__main__":
    main()
