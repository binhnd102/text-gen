import tensorflow as tf
from typing import Text
from preprocessing import clean_str, clean_input_text
from data import get_data, get_dataset
from text_vectorizer import TextVectorizer
from model import TextGenModel, TextGenServingModel
from callbacks import get_callbacks
from one_model import OneStep


text = get_data()
vectorizer = TextVectorizer.init_from_text(text)
dataset = get_dataset(vectorizer, text, 1)

x, y = next(dataset.take(1).as_numpy_iterator())

for i in tf.strings.reduce_join(vectorizer.chars_from_ids(x), axis=1).numpy():
    print(i.decode('utf-8'))

for i in tf.strings.reduce_join(vectorizer.chars_from_ids(y), axis=1).numpy():
    print(i.decode('utf-8'))



embedding_dim = 256
rnn_units = 1024
checkpoint_dir = './checkpoints/'
epoch = 50


VOCAB_SIZE = len(vectorizer.get_vocab())
model = TextGenModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)


callbacks = get_callbacks(checkpoint_dir + 'TextGenCkpt/')
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
history = model.fit(dataset, epochs=epoch, callbacks=callbacks)
model.save_weights(checkpoint_dir + 'best_weight')

serving_model = TextGenServingModel(
    model.embedding,
    model.gru,
    model.dense
)
x, y = next(dataset.take(1).as_numpy_iterator())
states = tf.zeros((x.shape[0], 1024))
output, _ = serving_model((x, states))
print(output)
serving_model.save('saved_model/serving_model')
