"""Greedy CTC decoding: model logits -> text."""

import tensorflow as tf

from src.vocab import decode_indices


def greedy_decode(y_pred):
    """y_pred: (batch, time, num_classes) softmax output -> list[str] predictions."""
    input_len = tf.ones(tf.shape(y_pred)[0], dtype="int32") * tf.shape(y_pred)[1]
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)
    decoded = decoded[0]  # (batch, max_decoded_len), padded with -1

    # ctc_decode pads with -1; map those to 0 (the same index used for
    # pad/OOV in vocab.py) since num_to_char maps 0 -> "" anyway.
    decoded = tf.where(decoded == -1, tf.zeros_like(decoded), decoded)
    return decode_indices(decoded)
