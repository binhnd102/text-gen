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
    'bán',
    'mua',
    'cần mua',
    'cần bán',
    'thanh lý',
    'thanh lí',
    'cần thanh lí',
    'cần thanh lý',
    'mình',
    'dọn',
    'xả kho',
    'chính chủ',
    'pass',
    'cho',
    'tặng',
    'cần',
    'dư',
    'gấp',
    'dư',
    'nhà',
    'tôi',
    'cầm',
    'em',
    'đổi',
    'ra',
    'gấp',
    'nhanh',
    'do',
    'lên',
    'không',
    'kẹt',
    'tiền',
    'giao',
    'hon',
    'muốn',
    'tay',
    'dư',
    'nhà',
    'không',
    'tôi',
    'thể',
    'ko',
    'em',
    'gấp',
    'ra',
    'để',
    'sale',
    'muốn',
    'con',
    'thank',
    'hàng',
    'do',
    'ít',
    'tìm',
    'bé',
    'có',
    'nhanh',
    'lên',
    'đổi',
    'chật',
    'còn',
    'giá',
    'fix',
    'lại',
    'rẻ',
    'chạy',
    'khong',
    'xả',
    'thừa',
    'cháu',
    'cuộc',
    'chuyên',
    'về',
    'vì',
    'xếp',
    'kho',
    'chuyển',
    'một',
    'dư',
    'nhà',
    'thừa',
    'em',
    'không',
    'bò',
    'se',
    'ko',
    'cap',
    'do',
    'có',
    'gấp',
    'nhanh',
    'xả',
    'chuyên',
    'có',
    'lại',
    'nhiều',
    'giá',
    'nghỉ',
    'màu',
    'còn',
    'mới',
    'để',
    'ít',
    'đát',
    'giày',
    'chỉ',
    'do',
    'rẻ',
    'về',
    '<<',
    'sale',
    'để',
    'lên',
    'giá',
    'dư',
    'cái',
    'em',
    'lô',
    'shop',
    'không',
    'xách',
    'hộp',
    'có',
    'còn',
    'ko',
    'mới',
    'saleoff',
    'nhượng',
    'giảm',
    'sang',
    'nhà',
    'cốp',
    'free',
    'hai',
    'kho',
    'xả',
    'sưu',
    'tuyển',
    'nhượng',
    'em',
    'tìm',
    'tất',
    'nghệ',
    'conbo',
    'em',
    'dư',
    'nhanh',
    'tìm',
    'lên',
    'dư',
    'kẹt',
    'thừa',
    'nâng',
    'nhà',
    'xả',
    'lên',
    'nhượng',
    'muốn'
    ]

pattern = r"^{}".format('(' + '|'.join(filter_words) + ')')


def clean_input_text(text):
    text = clean_str(text)
    text = re.sub(pattern, '', text).strip()
    return text