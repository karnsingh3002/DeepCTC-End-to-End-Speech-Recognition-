"""Training entry point.

Usage:
    python -m src.train --epochs 100 --batch_size 32 --lr 1e-4
    python -m src.train --epochs 5 --subset 200   # smoke test on a small subset
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    ASSETS_DIR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SPEC_FEATURE_DIM,
)
from src.dataset import build_dataset, train_val_split
from src.decode import greedy_decode
from src.losses import ctc_loss
from src.model import build_model
from src.vocab import VOCAB_SIZE, decode_indices


class PredictionSampler(tf.keras.callbacks.Callback):
    """Prints a few greedy-decoded validation predictions after each epoch."""

    def __init__(self, val_ds, num_samples=2):
        super().__init__()
        self.val_batch = next(iter(val_ds.take(1)))
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        specs, labels = self.val_batch
        preds = self.model.predict(specs, verbose=0)
        pred_texts = greedy_decode(preds)
        true_texts = decode_indices(labels)
        print(f"\n--- Sample predictions (epoch {epoch + 1}) ---")
        for i in range(min(self.num_samples, len(pred_texts))):
            print(f"  true: {true_texts[i]!r}")
            print(f"  pred: {pred_texts[i]!r}")
        print("---------------------------------------\n")


def plot_loss_curve(history, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("CTC loss")
    plt.title("DeepCTC training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the DeepCTC acoustic model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, train/val on only this many examples each (smoke test).",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_df, val_df = train_val_split()
    if args.subset:
        train_df = train_df.head(args.subset).reset_index(drop=True)
        val_df = val_df.head(max(1, args.subset // 10)).reset_index(drop=True)
    print(f"Train examples: {len(train_df)}  Val examples: {len(val_df)}")

    train_ds = build_dataset(train_df, batch_size=args.batch_size, shuffle=True)
    val_ds = build_dataset(val_df, batch_size=args.batch_size, shuffle=False)

    model = build_model(input_dim=SPEC_FEATURE_DIM, output_dim=VOCAB_SIZE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=ctc_loss)
    model.summary()

    checkpoint_path = os.path.join(args.checkpoint_dir, "deepctc.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
        PredictionSampler(val_ds),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,  # train_ds is already shuffled by build_dataset()
    )

    plot_loss_curve(history, os.path.join(ASSETS_DIR, "loss_curve.png"))
    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
