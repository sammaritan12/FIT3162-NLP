import pickle
import time
from sys import argv

from sklearn.preprocessing import normalize

import config
from classifier.english_classifier import english_classifier
from classifier.spanish_classifier import spanish_classifier

if __name__ == '__main__':
    '''
    Creates English, Spanish or both classifiers using the extracted features (extract_feature.py) found in directory placed in config.py
    Can be run as:
    python create_classifier.py <spanish/english>
    If no language is specified, it is assumed both will be created
    '''
    t1 = time.time()
    language = config.BOTH

    # Check what language is to be used
    if len(argv) > 1:
        if argv[1].lower() == config.EN_NAME.lower():
            language = config.ENGLISH
        elif argv[1].lower() == config.SP_NAME.lower():
            language = config.SPANISH

    # FEATURE SETS LOADING FROM FILE #

    # Loading English Features
    if language >= 0:
        with open(config.en_authors_path, 'rb') as fid:
            en_authors = pickle.load(fid)

        with open(config.en_char_ngram_feature_set_path, 'rb') as fid:
            en_char_ngram_feature_set = pickle.load(fid)

        # Features below can be commented as they are not necessary for most accurate classifier

        with open(config.en_word_ngram_feature_set_path, 'rb') as fid:
            en_word_ngram_feature_set = pickle.load(fid)

        with open(config.en_avg_sentence_length_feature_set_path, 'rb') as fid:
            en_avg_sentence_length_feature_set = pickle.load(fid)

        with open(config.en_punctuation_feature_set_path, 'rb') as fid:
            en_punctuation_feature_set = pickle.load(fid)

    # Loading Spanish Features
    if language <= 0:
        with open(config.sp_authors_path, 'rb') as fid:
            sp_authors = pickle.load(fid)

        with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
            sp_char_ngram_feature_set = pickle.load(fid)

        # Features below can be commented as they are not necessary for most accurate classifier

        with open(config.sp_word_ngram_feature_set_path, 'rb') as fid:
            sp_word_ngram_feature_set = pickle.load(fid)

        with open(config.sp_avg_sentence_length_feature_set_path, 'rb') as fid:
            sp_avg_sentence_length_feature_set = pickle.load(fid)

        with open(config.sp_punctuation_feature_set_path, 'rb') as fid:
            sp_punctuation_feature_set = pickle.load(fid)

    # FEATURE NORMALIZATION AND NORMALIZATION #
    t0 = time.time()
    
    # English Features
    if language >= 0:
        en_training_feature_set = []

        # Uncomment this if desired to test other classifiers
        # Assembling them such that they look like x
        # for i in range(len(en_authors)):
        #     en_training_feature_set.append(
        #         [en_avg_sentence_length_feature_set[i]] + en_char_ngram_feature_set[i] \
        #         + en_word_ngram_feature_set[i] + en_punctuation_feature_set[i])

        # This test feature set is the most accurate, others worsen the accuracy
        en_training_feature_set = en_char_ngram_feature_set

        en_training_feature_set_normalised = normalize(en_training_feature_set, norm=config.normalization_type)

    # Spanish Features
    if language <= 0:
        sp_training_feature_set = []

        # Uncomment this if desired to test other classifiers
        # Assembling them such that they look like x
        # for i in range(len(sp_authors)):
        #     sp_training_feature_set.append(
        #         [sp_avg_sentence_length_feature_set[i]] + sp_char_ngram_feature_set[i] \
        #         + sp_word_ngram_feature_set[i] + sp_punctuation_feature_set[i])

        # This test feature set is the most accurate, others worsen the accuracy
        sp_training_feature_set = sp_char_ngram_feature_set

        sp_training_feature_set_normalised = normalize(sp_training_feature_set, norm=config.normalization_type)

    print("Zipping and Normalisation Time:", time.time() - t0)
    t0 = time.time()

    # CLASSIFIER TRAINING AND FILE SAVING #
    # English Classifier
    if language >= 0:
        en_classifier = english_classifier(en_training_feature_set_normalised, en_authors)

        # Serialise classifier
        with open(config.en_classifier_path, 'wb') as fid:
            pickle.dump(en_classifier, fid)
    
    # Spanish Classifier
    if language <= 0:
        sp_classifier = spanish_classifier(sp_training_feature_set_normalised, sp_authors)

        # Serialise classifier
        with open(config.sp_classifier_path, 'wb') as fid:
            pickle.dump(sp_classifier, fid)

    print("classifier Fit and File Saving Time:", time.time() - t0)

    # Comment success
    if language >= 0:
        print(config.EN_NAME, "Classifier created in", time.time() - t1, "seconds.")
    if language <= 0:
        print(config.SP_NAME, "Classifier created in", time.time() - t1, "seconds.")
