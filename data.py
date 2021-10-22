from preprocessing import clean_input_text
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd

BUFFER_SIZE = 10000

def pad_sequence(sequence):
    return keras.preprocessing.sequence.pad_sequences(sequence, 
                                                    maxlen=60, 
                                                    dtype='int32', 
                                                    padding='post', 
                                                    truncating='post', 
                                                    value=0.0)

def get_dataset(vectorizer, text, batch_size):
    split_ids = []
    cur = 0
    for sent in text[:-1]:
        split_ids.append(cur + len(sent))
        cur += len(sent)

    chars = tf.strings.unicode_split(''.join(text), input_encoding='UTF-8').numpy()
    ids = vectorizer.ids_from_chars(chars)

    X = []
    Y = []
    split_tokens = np.split(ids, indices_or_sections=split_ids)
    for tokens in split_tokens:
        X.append(tokens[:-1])
        Y.append(tokens[1:])
    X = pad_sequence(X)
    Y = pad_sequence(Y)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = (
        dataset
        # .shuffle(BUFFER_SIZE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return dataset


def get_data():
    df = pd.read_csv('data/sample_data.csv')
    text = df.subject.apply(clean_input_text).values.tolist()
    titles = []
    for title in text[:2]:
        titles.append('[START]{}[END]'.format(title))
    return titles