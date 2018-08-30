import pickle
import time
from sys import argv

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

import config
from classifier.english_classifier import english_classifier

if __name__ == '__main__':
    t1 = time.time()

    # 0 == both
    # 1 == english
    # -1 == spanish
    language = 0

    if len(argv) > 1:
        if argv[1].lower() == 'english':
            language = 1
        elif argv[1].lower() == 'spanish':
            language = -1

    # FEATURE SETS LOADING FROM FILE #

    # Loading English Features
    if language >= 0:
        with open(config.en_authors, 'rb') as fid:
            en_authors = pickle.load(fid)

        with open(config.en_char_ngram_feature_set_path, 'rb') as fid:
            en_char_ngram_feature_set = pickle.load(fid)

        with open(config.en_word_ngram_feature_set_path, 'rb') as fid:
            en_word_ngram_feature_set = pickle.load(fid)

        with open(config.en_avg_sentence_length_feature_set_path, 'rb') as fid:
            en_avg_sentence_length_feature_set = pickle.load(fid)

    # Loading Spanish Features
    if language <= 0:
        with open(config.sp_authors, 'rb') as fid:
            sp_authors = pickle.load(fid)

        with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
            sp_char_ngram_feature_set = pickle.load(fid)

        with open(config.sp_word_ngram_feature_set_path, 'rb') as fid:
            sp_word_ngram_feature_set = pickle.load(fid)

        with open(config.sp_avg_sentence_length_feature_set_path, 'rb') as fid:
            sp_avg_sentence_length_feature_set = pickle.load(fid)

    # FEATURE NORMALIZATION AND NORMALIZATION #
    t0 = time.time()
    
    # English Features
    if language >= 0:
        en_training_feature_set = []

        # Assembling them such that they look like x
        for i in range(len(en_authors)):
            en_training_feature_set.append(
                [en_avg_sentence_length_feature_set[i]] + en_char_ngram_feature_set[i] + en_word_ngram_feature_set[i])
        
        en_training_feature_set_normalised = normalize(en_training_feature_set, norm=config.normalization_type)

    # Spanish Features
    if language <= 0:
        sp_training_feature_set = []

        # Assembling them such that they look like x
        for i in range(len(sp_authors)):
            sp_training_feature_set.append(
                [sp_avg_sentence_length_feature_set[i]] + sp_char_ngram_feature_set[i] + sp_word_ngram_feature_set[i])

        sp_training_feature_set_normalised = normalize(sp_training_feature_set, norm=config.normalization_type)

    print("Zipping and Normalisation Time:", time.time() - t0)
    t0 = time.time()

    # CLASSIFIER TRAINING AND FILE SAVING #
