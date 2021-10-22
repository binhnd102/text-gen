import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import pickle

class TextVectorizer:
    @classmethod
    def init_from_text(cls, text):
        self = cls.__new__(cls)
        vocab = sorted(set(' '.join(text)))
        print(f'{len(vocab)} unique characters')
        self._ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), 
                                                          mask_token='')
        self._chars_from_ids = preprocessing.StringLookup(vocabulary=list(vocab), 
                                                          invert=True,
                                                          mask_token='')
        self.vocab = self._chars_from_ids.get_vocabulary()
        return self
    
    @classmethod
    def init_from_vocab(cls, filepath):
        self = cls.__new__(cls)
        vocab = pickle.load(open(filepath, 'rb'))
        print(f'{len(vocab)} unique characters')
        self._ids_from_chars = preprocessing.StringLookup(vocabulary=vocab[2:], 
                                                          mask_token='')
        self._chars_from_ids = preprocessing.StringLookup(vocabulary=vocab[2:], 
                                                          invert=True,
                                                          mask_token='')
        self.vocab = self._chars_from_ids.get_vocabulary()
        return self
    
    def ids_from_chars(self, chars):
        return self._ids_from_chars(chars)
    
    def chars_from_ids(self, ids):
        return self._chars_from_ids(ids)
    
    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self._chars_from_ids(ids), axis=-1)
    
    def get_vocab(self):
        return self._chars_from_ids.get_vocabulary()
    
    def save(self, filepath):
        pickle.dump(self.vocab, open(filepath, 'wb'))