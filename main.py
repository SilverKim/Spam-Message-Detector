###############################################################################
# Start
###############################################################################

import glob
import numpy as np
import utils

nonspam_path = glob.glob('data/ham/*')
spam_path = glob.glob('data/spam/*')

###############################################################################
# Split data into training and test sets
###############################################################################

from sklearn.model_selection import train_test_split

arr = np.array(train_test_split(nonspam_path), dtype="object")

nonspam_train = np.array(arr[0])
nonspam_test = np.array(arr[1])

spam_sample = np.array(train_test_split(spam_path), dtype="object")

spam_train = np.array(spam_sample[0])
spam_test = np.array(spam_sample[1])

nonspam_train_label = [0] * nonspam_train.shape[0]
spam_train_label = [1] * spam_train.shape[0]
x_train = np.concatenate((nonspam_train, spam_train))
y_train = np.concatenate((nonspam_train_label, spam_train_label))

nonspam_test_label = [0] * nonspam_test.shape[0]
spam_test_label = [1] * spam_test.shape[0]
x_test = np.concatenate((nonspam_test, spam_test))
y_test = np.concatenate((nonspam_test_label, spam_test_label))

train_shuffle_index = np.random.permutation(np.arange(0, x_train.shape[0]))
test_shuffle_index = np.random.permutation(np.arange(0, x_test.shape[0]))

x_train = x_train[train_shuffle_index]
y_train = y_train[train_shuffle_index]

x_test = x_test[test_shuffle_index]
y_test = y_test[test_shuffle_index]

x_train = utils.get_email_content_bulk(x_train)
x_test = utils.get_email_content_bulk(x_test)

x_train, y_train = utils.remove_null(x_train, y_train)
x_test, y_test = utils.remove_null(x_test, y_test)

###############################################################################
# Clean data (part 1)
###############################################################################

x_train = [utils.clean_up_pipeline(o) for o in x_train]
x_test = [utils.clean_up_pipeline(o) for o in x_test]

###############################################################################
# Clean data (part 2)
###############################################################################

from nltk.tokenize import word_tokenize

# Tokenization are taking slightly longer to process
x_train = [word_tokenize(o) for o in x_train]
x_test = [word_tokenize(o) for o in x_test]

x_train = [utils.clean_token_pipeline(o) for o in x_train]
x_test = [utils.clean_token_pipeline(o) for o in x_test]

###############################################################################
# Feature extraction - TfidfVectorizer
###############################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()
raw = [' '.join(o) for o in x_train]
v.fit(raw)

def features(raw_tokenize_data):
    raw = [' '.join(o) for o in raw_tokenize_data]
    return v.transform(raw)

x_train_features = features(x_train)
x_test_features = features(x_test)

###############################################################################
# Feature extraction - CountVectorizer
###############################################################################

# from sklearn.feature_extraction.text import CountVectorizer
#
# v = CountVectorizer()
# raw = [' '.join(o) for o in x_train]
# v.fit(raw)
#
# x_train_features = features(x_train)
# x_test_features = features(x_test)

###############################################################################
# Training classifier - Gaussian Naive Bayes
###############################################################################

from sklearn.naive_bayes import GaussianNB

c = GaussianNB()

c.fit(x_train_features.toarray(), y_train)

c.score(x_test_features.toarray(), y_test)

c.score(x_train_features.toarray(), y_train)

###############################################################################
# Error Analysis - Confusion Matrix
###############################################################################

from sklearn.metrics import confusion_matrix, precision_score, recall_score

y_predict = c.predict(x_test_features.toarray())

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predict)))

result = confusion_matrix(y_test, y_predict)
print(result)
