"""DeepSpeech2-style CNN + BiGRU acoustic model builder."""

import tensorflow as tf

from src.config import DROPOUT, RNN_LAYERS, RNN_UNITS

# Both conv layers use stride 2 along the time axis -> overall /4 downsampling.
CONV_TIME_STRIDES = (2, 2)
CONV_FREQ_STRIDES = (2, 2)


def compute_time_steps_after_conv_tf(time_steps):
    """Same as above, for use inside a tf.data graph (tf.Tensor of ints)."""
    time_steps = tf.cast(time_steps, tf.float32)
    for stride in CONV_TIME_STRIDES:
        time_steps = tf.math.ceil(time_steps / stride)
    return tf.cast(time_steps, tf.int32)


def build_model(input_dim, output_dim, rnn_units=RNN_UNITS, rnn_layers=RNN_LAYERS):
    """Build the CNN + BiGRU CTC acoustic model.

    Args:
        input_dim: number of frequency bins in the input spectrogram.
        output_dim: vocabulary size (blank class is added on top, so the
            final Dense layer has output_dim + 1 units).
        rnn_units: units per GRU direction.
        rnn_layers: number of stacked Bidirectional GRU layers.
    """
    input_spec = tf.keras.layers.Input(shape=(None, input_dim), name="spectrogram")
    x = tf.keras.layers.Reshape((-1, input_dim, 1))(input_spec)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=CONV_TIME_STRIDES[0:1] + CONV_FREQ_STRIDES[0:1],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_1_relu")(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=CONV_TIME_STRIDES[1:2] + CONV_FREQ_STRIDES[1:2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_2_relu")(x)

    # (batch, time', freq', channels) -> (batch, time', freq' * channels)
    shape = x.shape
    x = tf.keras.layers.Reshape((-1, shape[2] * shape[3]))(x)

    for i in range(1, rnn_layers + 1):
        recurrent = tf.keras.layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = tf.keras.layers.Bidirectional(recurrent, merge_mode="concat", name=f"bidirectional_{i}")(x)
        if i < rnn_layers:
            x = tf.keras.layers.BatchNormalization(name=f"bidirectional_{i}_bn")(x)
            x = tf.keras.layers.Dropout(DROPOUT, name=f"bidirectional_{i}_dropout")(x)

    x = tf.keras.layers.Dense(rnn_units * 2, name="dense_1")(x)
    x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dense_1_dropout")(x)

    output = tf.keras.layers.Dense(output_dim + 1, activation="softmax", name="output")(x)

    model = tf.keras.Model(input_spec, output, name="DeepCTC")
    return model


if __name__ == "__main__":
    from src.config import SPEC_FEATURE_DIM
    from src.vocab import VOCAB_SIZE

    m = build_model(input_dim=SPEC_FEATURE_DIM, output_dim=VOCAB_SIZE)
    m.summary()
