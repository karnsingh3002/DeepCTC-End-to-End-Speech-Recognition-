"""Character vocabulary and text encode/decode helpers."""

import tensorflow as tf

from src.config import VOCAB_CHARS

char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB_CHARS, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

VOCAB_SIZE = char_to_num.vocabulary_size()


def encode_text(text):
    """String (or scalar tf.string tensor) -> 1D int64 tensor of char indices."""
    text = tf.strings.lower(text)
    chars = tf.strings.unicode_split(text, input_encoding="UTF-8")
    return char_to_num(chars)


def decode_indices(indices):
    """1D (or batch) int tensor of char indices -> python string(s)."""
    chars = num_to_char(indices)
    text = tf.strings.reduce_join(chars, axis=-1)
    if text.shape.rank == 0:
        return text.numpy().decode("utf-8")
    return [t.decode("utf-8") for t in text.numpy()]


if __name__ == "__main__":
    print(f"Vocab size (incl. OOV/blank slot): {VOCAB_SIZE}")
    print(f"Vocabulary: {char_to_num.get_vocabulary()}")

    sample = "The quick brown fox, jumps!"
    encoded = encode_text(sample)
    decoded = decode_indices(encoded)
    print(f"Original : {sample}")
    print(f"Encoded  : {encoded.numpy().tolist()}")
    print(f"Decoded  : {decoded}")
    assert decoded == sample.lower().replace(",", "").replace("!", ""), (
        "round-trip mismatch (expected OOV chars like , and ! to be dropped)"
    )
