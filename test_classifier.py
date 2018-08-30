from datetime import datetime
import pickle
import time
from sys import argv

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

import config
from classifier.english_classifier import english_classifier


def cv_classifier_score(classifier, train, target):
    cv_score =  cross_val_score(classifier, train, target, cv=10)
    return "Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2)


def test_classifier_kernels(authors, features_normalized, features_text, language):
    """
    Creates a text file which outputs the results for english classifier kernel test.
    - authors, the authors within the classifier, the target file
    - features_normalized, normalized feature set to be used by the classifier
    - features_text, text of what is included in the feature set being passed
    """
    curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    lang_name = 'unknown'

    if language == config.ENGLISH:
        lang_name = 'en'
    elif language == config.SPANISH:
        lang_name = 'sp'

    output_file = open('./testing/' + lang_name + '_classifier_kernel_test_' + curr_time + '.txt', 'w+')
    output_file.write("TEST: English Classifier Kernels " + curr_time + '\n')
    output_file.write("Features:\n" + features_text +"\n")

    print("TEST: English Classifier Kernels " + curr_time)
    print("Features:\n" + features_text)

    # Linear SVC Kernel
    linear_svc = LinearSVC()
    linear_svc_score = cv_classifier_score(linear_svc, features_normalized, authors)
    output_file.write(linear_svc_score + '\n')
    print(linear_svc_score)

    # SVC Linear Kernel
    svc_linear = SVC(kernel='linear')
    svc_linear_score = cv_classifier_score(svc_linear, features_normalized, authors)
    output_file.write(svc_linear_score + '\n')
    print(svc_linear_score)

    # SVC Poly Kernel
    svc_poly = SVC(kernel='poly')
    svc_poly_score = cv_classifier_score(svc_poly, features_normalized, authors)
    output_file.write(svc_poly_score + '\n')
    print(svc_poly_score)

    # SVC rbf Kernel
    svc_rbf = SVC(kernel='rbf')
    svc_rbf_score = cv_classifier_score(svc_rbf, features_normalized, authors)
    output_file.write(svc_rbf_score + '\n')
    print(svc_rbf_score)

    # SVC Sigmoid Kernel
    svc_sigmoid = SVC(kernel='sigmoid')
    svc_sigmoid_score = cv_classifier_score(svc_sigmoid, features_normalized, authors)
    output_file.write(svc_sigmoid_score + '\n')
    print(svc_sigmoid_score)

    # SVC Precomputed Kernel
    svc_precomputed = SVC(kernel='precomputed')
    svc_precomputed_score = cv_classifier_score(svc_precomputed, features_normalized, authors)
    output_file.write(svc_precomputed_score + '\n')
    print(svc_precomputed_score)

    output_file.close()

    return


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
    # if language <= 0:
    #     with open(config.sp_authors, 'rb') as fid:
    #         sp_authors = pickle.load(fid)

    #     with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
    #         sp_char_ngram_feature_set = pickle.load(fid)

    #     with open(config.sp_word_ngram_feature_set_path, 'rb') as fid:
    #         sp_word_ngram_feature_set = pickle.load(fid)

    #     with open(config.sp_avg_sentence_length_feature_set_path, 'rb') as fid:
    #         sp_avg_sentence_length_feature_set = pickle.load(fid)

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
    # if language <= 0:
    #     sp_training_feature_set = []

    #     # Assembling them such that they look like x
    #     for i in range(len(sp_authors)):
    #         sp_training_feature_set.append(
    #             [sp_avg_sentence_length_feature_set[i]] + sp_char_ngram_feature_set[i] + sp_word_ngram_feature_set[i])

    #     sp_training_feature_set_normalised = normalize(sp_training_feature_set, norm=config.normalization_type)

    print("Zipping and Normalisation Time:", time.time() - t0)
    t0 = time.time()

    # CLASSIFIER TRAINING AND FILE SAVING #
