import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TextGenModel(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        mask = self.embedding.compute_mask(inputs)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training, mask=mask)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class TextGenServingModel(keras.Model):
    def __init__(self, embedding, gru, dense):
        super().__init__(self)
        self.embedding = embedding
        self.gru = gru
        self.dense = dense
        self.built = True

    @tf.function(
        input_signature=[
            (tf.TensorSpec([None, 60], dtype=tf.int32), tf.TensorSpec([None, 1024]))
        ]
    )
    def call(self, inputs):
        x, states = inputs
        mask = self.embedding.compute_mask(x)
        x = self.embedding(x, training=False)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=False, mask=mask)
        x = self.dense(x, training=False)
        perm_x = tf.transpose(x, perm=[1, 0, 2])
        # trim the mask and revert
        x = tf.transpose(perm_x[mask[0]], perm=[1, 0, 2])
        return x, states
