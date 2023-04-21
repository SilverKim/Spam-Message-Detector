import email
import re
import string

import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

# Extracting email content

def get_email_content(path):
    f = open(path, encoding='latin1')
    try:
        msg = email.message_from_file(f)
        for i in msg.walk():
            if i.get_content_type() == 'text/plain':
                return i.get_payload()  # prints the raw text
    except Exception as e:
        print(e)

def get_email_content_bulk(paths):
    contents = [get_email_content(o) for o in paths]
    return contents

def remove_null(datas, labels):
    not_null_idx = [i for i, o in enumerate(datas) if o is not None]
    return np.array(datas)[not_null_idx], np.array(labels)[not_null_idx]

# Cleaning

def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)

def to_lower(word):
    s = word.lower()
    return s

def remove_number(word):
    s = re.sub(r'\d+', '', word)
    return s

def remove_punctuation(word):
    s = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return s

def remove_whitespace(word):
    s = word.strip()
    return s

def replace_newline(word):
    return word.replace('\n', '')

def clean_up_pipeline(s):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,
                      remove_whitespace]
    for o in cleaning_utils:
        s = o(s)
    return s

# Tokenization

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_stop_words(words):
    s = [i for i in words if i not in ENGLISH_STOP_WORDS]
    return s

def word_stemmer(words):
    return [stemmer.stem(o) for o in words]

def word_lemmatizer(words):
    return [lemmatizer.lemmatize(o) for o in words]

def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words, word_stemmer, word_lemmatizer]
    for o in cleaning_utils:
        s = o(words)
    return s
