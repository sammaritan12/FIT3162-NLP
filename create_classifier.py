import pickle
import config
import time
from sklearn.preprocessing import normalize
from classifier.english_classifier import *

if __name__ == '__main__':
    # FEATURE SETS LOADING FROM FILE #

    with open(config.en_authors, 'rb') as fid:
        authors = pickle.load(fid)

    with open(config.en_char_ngram_feature_set_path, 'rb') as fid:
        en_char_ngram_feature_set = pickle.load(fid)

    with open(config.en_word_ngram_feature_set_path, 'rb') as fid:
        en_word_ngram_feature_set = pickle.load(fid)

    with open(config.en_avg_sentence_length_feature_set_path, 'rb') as fid:
        en_avg_sentence_length_feature_set = pickle.load(fid)

    # FEATURE NORMALIZATION #
    t0 = time.time()

    training_feature_set = []

    # Assembling them such that they look like x
    for i in range(len(en_avg_sentence_length_feature_set)):
        training_feature_set.append(
            [en_avg_sentence_length_feature_set[i]] + en_char_ngram_feature_set[i] + en_word_ngram_feature_set[i])

    print("Zipping Features Time:", time.time() - t0)

    training_feature_set_normalised = normalize(training_feature_set, norm=config.normalization_type)

    print("Normalization Time:", time.time() - t0)
    t0 = time.time()

    # CLASSIFIER TRAINING #
    en_classifier = english_classifier(training_feature_set_normalised, authors)

    print("classifier Fit Time:", time.time() - t0)

    # Save the classifier
    with open(config.en_classifier_path, 'wb') as fid:
        pickle.dump(en_classifier, fid)
