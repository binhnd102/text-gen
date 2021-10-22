import tensorflow as tf
from tensorflow import keras
from typing import Text
from preprocessing import clean_str, clean_input_text
from data import get_data, get_dataset
from text_vectorizer import TextVectorizer
from model import TextGenModel, TextGenServingModel
from callbacks import get_callbacks
from one_model import OneStep
from time import time


text = get_data()
vectorizer = TextVectorizer.init_from_text(text)
dataset = get_dataset(vectorizer, text, 2)

serving_model = keras.models.load_model("saved_model/serving_model")
x, y = next(dataset.take(1).as_numpy_iterator())
states = tf.zeros((x.shape[0], 1024))
output, _ = serving_model((x, states))


one_step_model = OneStep(serving_model, vectorizer)
# one_step_model = keras.models.load_model("saved_model/one_step")


def generate_sent():
    next_char = tf.constant(["[START]"] * 2)
    states = tf.zeros((next_char.shape[0], 1024))
    result = [next_char]
    for _ in tf.range(20):
        next_char, states = one_step_model((next_char, states))
        result.append(next_char)

    result = tf.strings.reduce_join(result, axis=0)
    return result


@tf.function(autograph=True)
def f():
    next_char = tf.constant(["[START]"] * 2)
    states = tf.zeros((next_char.shape[0], 1024))
    result = tf.expand_dims(next_char, 1)
    for i in tf.range(20):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[
                (result, tf.TensorShape([2, None])),
                (
                    next_char,
                    tf.TensorShape(
                        [
                            None,
                        ]
                    ),
                ),
                (states, tf.TensorShape([None, 1024])),
            ]
        )
        next_char, states = one_step_model((next_char, states))
        result = tf.concat((result, tf.reshape(next_char, (2, 1))), 1)
    final_result = tf.strings.reduce_join(result, 1)
    return final_result


begin = time()
for i in range(10):
    print(f())
print(time() - begin)
print()
begin = time()
for i in range(10):
    print(generate_sent())
print(time() - begin)
print()
begin = time()
for i in range(10):
    print(f())
print(time() - begin)


# one_step_model.save("saved_model/one_step")
