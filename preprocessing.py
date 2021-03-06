import re
from pyvi import ViUtils
import unicodedata


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

def remove_accent(string):
    return ViUtils.remove_accents(string).decode('utf-8')


def clean_str(string, remove_accents=False):
    """
....Tokenization/string cleaning for all datasets except for SST.
....Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
...."""

    # string = re.sub(r"[^A-Za-z0-9()/,!?\'\`]", " ", string)

    string = string.lower()
    string = unicodedata.normalize('NFKD', string)
    string = re.sub(r"[0](\d|\u002E|\u0020){9,15}", ' PHONENUMBER ',
                    string)
    string = \
        string.strip("/ . , ; / ( ) / { }[\" ] # $ @ ! % ^ & * +/-\ _ /"
                     )
    string = string.replace('`', r"")
    string = string.replace(r"", r"")
    string = string.replace('  ', ' ')
    string = string.replace('(', ' ')
    string = string.replace(')', ' ')
    string = string.replace('--', ' ')
    string = string.replace('[', ' ')
    string = string.replace(']', ' ')
    string = string.replace('{', ' ')
    string = string.replace('}', ' ')
    string = string.replace('\xe2\x80\xa2', ' ')
    string = string.replace('...', ' ')
    string = string.replace('..', ' ')
    string = emoji_pattern.sub(r"", string)
    string = string.strip(' \t\n\r')
    string = string.lstrip().rstrip()
    string = string.replace('ch\xe1\xbb\xa3 t\xe1\xbb\x91t',
                            'ch\xe1\xbb\xa3_t\xe1\xbb\x91t')
    string = string.replace('cho tot', 'cho_tot')
    # string = re.sub(r",", ' , ', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"!", ' ', string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", ' ', string)
    string = re.sub(r"-", ' ', string)

    # string = re.sub(r"(=>|\|?|_|;){1,}"," ",string)

    string = string.replace('=>', ' ')
    string = re.sub(r"\u005C", ' ', string)
    string = string.replace('?', ' ')
    string = string.replace('_', ' ')
    string = string.replace(';', ' ')
    string = re.sub('\xe2\x80\x94{1,}', ' ', string)
    string = re.sub(r"#{1,}", ' ', string)
    string = re.sub(r":{1,}", ' ', string)
    string = string.replace('^', ' ')
    string = re.sub(r">{1,}", ' ', string)
    string = string.replace('+', ' ')
    string = string.replace('*', ' ')
    string = string.replace('"', ' ')
    string = string.replace('=', ' ')
    string = string.replace('\xe2\x80\x93', ' ')
    string = string.replace(u'\xa0', u' ')
    string = string.replace('. .', ' ')
    # string = re.sub(r" {2,}", ' ', string)
    if remove_accents:
        string = string.replace(r",", ' ')
        string = re.sub(r"/{1,}", ' ', string)
        string = string.replace('.', ' ')
        string = string.replace(r"-", ' ')
        string = string.replace('/', ' ')
        string = ViUtils.remove_accents(string).decode('utf-8')
    string = re.sub(r" {2,}", ' ', string)
    return string.strip()


import numpy as np
import pandas as pd
import pickle
import re
import pandas as pd
import os
import shutil
import glob
from pathlib import Path



filter_words = [
    'b??n',
    'mua',
    'c???n mua',
    'c???n b??n',
    'thanh l??',
    'thanh l??',
    'c???n thanh l??',
    'c???n thanh l??',
    'm??nh',
    'd???n',
    'x??? kho',
    'ch??nh ch???',
    'pass',
    'cho',
    't???ng',
    'c???n',
    'd??',
    'g???p',
    'd??',
    'nh??',
    't??i',
    'c???m',
    'em',
    '?????i',
    'ra',
    'g???p',
    'nhanh',
    'do',
    'l??n',
    'kh??ng',
    'k???t',
    'ti???n',
    'giao',
    'hon',
    'mu???n',
    'tay',
    'd??',
    'nh??',
    'kh??ng',
    't??i',
    'th???',
    'ko',
    'em',
    'g???p',
    'ra',
    '?????',
    'sale',
    'mu???n',
    'con',
    'thank',
    'h??ng',
    'do',
    '??t',
    't??m',
    'b??',
    'c??',
    'nhanh',
    'l??n',
    '?????i',
    'ch???t',
    'c??n',
    'gi??',
    'fix',
    'l???i',
    'r???',
    'ch???y',
    'khong',
    'x???',
    'th???a',
    'ch??u',
    'cu???c',
    'chuy??n',
    'v???',
    'v??',
    'x???p',
    'kho',
    'chuy???n',
    'm???t',
    'd??',
    'nh??',
    'th???a',
    'em',
    'kh??ng',
    'b??',
    'se',
    'ko',
    'cap',
    'do',
    'c??',
    'g???p',
    'nhanh',
    'x???',
    'chuy??n',
    'c??',
    'l???i',
    'nhi???u',
    'gi??',
    'ngh???',
    'm??u',
    'c??n',
    'm???i',
    '?????',
    '??t',
    '????t',
    'gi??y',
    'ch???',
    'do',
    'r???',
    'v???',
    '<<',
    'sale',
    '?????',
    'l??n',
    'gi??',
    'd??',
    'c??i',
    'em',
    'l??',
    'shop',
    'kh??ng',
    'x??ch',
    'h???p',
    'c??',
    'c??n',
    'ko',
    'm???i',
    'saleoff',
    'nh?????ng',
    'gi???m',
    'sang',
    'nh??',
    'c???p',
    'free',
    'hai',
    'kho',
    'x???',
    's??u',
    'tuy???n',
    'nh?????ng',
    'em',
    't??m',
    't???t',
    'ngh???',
    'conbo',
    'em',
    'd??',
    'nhanh',
    't??m',
    'l??n',
    'd??',
    'k???t',
    'th???a',
    'n??ng',
    'nh??',
    'x???',
    'l??n',
    'nh?????ng',
    'mu???n'
    ]

pattern = r"^{}".format('(' + '|'.join(filter_words) + ')')


def clean_input_text(text):
    text = clean_str(text)
    text = re.sub(pattern, '', text).strip()
    return text