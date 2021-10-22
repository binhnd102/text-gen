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
        x = inputs

        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
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
        x = self.embedding(x, training=False)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=False)
        x = self.dense(x, training=False)
        return x, states
