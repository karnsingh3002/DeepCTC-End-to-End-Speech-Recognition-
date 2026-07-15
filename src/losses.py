"""CTC loss for the DeepCTC acoustic model."""

import tensorflow as tf


def ctc_loss(y_true, y_pred):
    """CTC loss.

    y_true: (label, label_length) tuple —
        label: (batch, max_label_len) int label indices, 0-padded.
        label_length: (batch,) true (pre-padding) length of each label, computed
            in dataset.py before padding. Using the padded batch's non-zero count
            instead would undercount whenever a real character (not just pad)
            maps to index 0, e.g. an out-of-vocabulary character.
    y_pred: (batch, time, num_classes) softmax output.
    """
    label, label_length = y_true
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    label_length = tf.reshape(tf.cast(label_length, dtype="int64"), (-1, 1))

    loss = tf.keras.backend.ctc_batch_cost(label, y_pred, input_length, label_length)
    return loss
