import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_paddings(pad_value):
    indices = [[1, 1]]  # A list of coordinates to update.
    values = [pad_value]  # A list of values corresponding to the respective
    # coordinate in indices.

    shape = [2, 2]  # The shape of the corresponding dense tensor, same as `c`.

    delta = tf.SparseTensor(indices, values, shape)
    delta = tf.sparse.to_dense(delta)
    return delta


class OneStep(tf.keras.Model):
    def __init__(self, model, vectorizer, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = vectorizer._chars_from_ids
        self.ids_from_chars = vectorizer._ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["", "[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(self.ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def call(self, inputs):
        inputs, states = inputs

        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, "UTF-8").to_tensor()
        input_ids = self.ids_from_chars(input_chars)
        maxlen = tf.math.reduce_max(
            tf.math.reduce_sum(tf.cast(input_ids != 0, tf.int32), axis=1)
        )
        padding = tf.math.reduce_max([0, 60 - maxlen])
        paddings = create_paddings(padding)
        padded = tf.pad(input_ids, paddings, mode="CONSTANT", constant_values=0)
        padded = tf.cast(padded, tf.int32)
        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model((padded, states))
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
