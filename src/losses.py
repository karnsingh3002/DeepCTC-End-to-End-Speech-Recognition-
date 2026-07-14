"""CTC loss for the DeepCTC acoustic model."""

import tensorflow as tf


def ctc_loss(y_true, y_pred):
    """CTC loss.

    y_true: (batch, max_label_len) int label indices, 0-padded.
    y_pred: (batch, time, num_classes) softmax output.
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    # label_length = count of non-zero (non-pad) positions per example.
    label_mask = tf.cast(tf.math.not_equal(y_true, 0), dtype="int64")
    label_length = tf.reduce_sum(label_mask, axis=1, keepdims=True)

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
