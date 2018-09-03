from datetime import datetime
import pickle
import time
from sys import argv
from itertools import combinations

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

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
    curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss-%fms")
    lang_name = 'unknown'

    if language == config.ENGLISH:
        lang_name = 'en'
    elif language == config.SPANISH:
        lang_name = 'sp'

    output_file = open('./testing/' + lang_name + '_classifier_kernel_test_' + curr_time + '.txt', 'a+')
    output_file.write("TEST: English Classifier Kernels " + curr_time + '\n')
    output_file.write("Features:\n" + features_text +"\n")

    print("TEST: English Classifier Kernels " + curr_time)
    print("Features:\n" + features_text)

    output_file.write('Using L2 Normalisation, Least Squares\n')
    print('Using', config.normalization_type.upper(), 'Normalisation\n')

    # Linear SVC Kernel
    linear_svc = LinearSVC()
    linear_svc_score = cv_classifier_score(linear_svc, features_normalized, authors)
    output_file.write("Linear SVC " + linear_svc_score + '\n')
    print("Linear SVC", linear_svc_score)

    # SVC Linear Kernel
    svc_linear = SVC(kernel='linear')
    svc_linear_score = cv_classifier_score(svc_linear, features_normalized, authors)
    output_file.write("SVC with Linear Kernel " + svc_linear_score + '\n')
    print("SVC with Linear Kernel", svc_linear_score)

    # SVC Poly Kernel
    svc_poly = SVC(kernel='poly')
    svc_poly_score = cv_classifier_score(svc_poly, features_normalized, authors)
    output_file.write("SVC with Poly Kerne l" + svc_poly_score + '\n')
    print("SVC with Poly Kernel", svc_poly_score)

    # SVC rbf Kernel
    svc_rbf = SVC(kernel='rbf')
    svc_rbf_score = cv_classifier_score(svc_rbf, features_normalized, authors)
    output_file.write("SVC with rbf Kernel " + svc_rbf_score + '\n')
    print("SVC with rbf Kernel", svc_rbf_score)

    # SVC Sigmoid Kernel
    svc_sigmoid = SVC(kernel='sigmoid')
    svc_sigmoid_score = cv_classifier_score(svc_sigmoid, features_normalized, authors)
    output_file.write("SVC with Sigmoid Kernel " + svc_sigmoid_score + '\n')
    print("SVC with Sigmoid Kernel", svc_sigmoid_score)

    # Gaussian Naive Bayes
    nb_gaussian = GaussianNB()
    nb_gaussian_score = cv_classifier_score(nb_gaussian, features_normalized, authors)
    output_file.write("Gaussian Naive Bayes " + nb_gaussian_score + '\n')
    print("Gaussian Naive Bayes", nb_gaussian_score)

    # Bernoulli Naive Bayes
    nb_bernoulli = BernoulliNB()
    nb_bernoulli_score = cv_classifier_score(nb_bernoulli, features_normalized, authors)
    output_file.write("Bernoulli Naive Bayes " + nb_bernoulli_score + '\n')
    print("Bernoulli Naive Bayes", nb_bernoulli_score)

    # Multinomial Naive Bayes
    nb_multinomial = MultinomialNB()
    nb_multinomial_score = cv_classifier_score(nb_multinomial, features_normalized, authors)
    output_file.write("Multinomial Naive Bayes" + nb_multinomial_score + '\n')
    print("Multinomial Naive Bayes", nb_multinomial_score, '\n')

    output_file.close()


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
        with open(config.en_authors_path, 'rb') as fid:
            en_authors = pickle.load(fid)

        with open(config.en_char_ngram_feature_set_path, 'rb') as fid:
            en_char_ngram_feature_set = pickle.load(fid)

        with open(config.en_word_ngram_feature_set_path, 'rb') as fid:
            en_word_ngram_feature_set = pickle.load(fid)

        with open(config.en_avg_sentence_length_feature_set_path, 'rb') as fid:
            en_avg_sentence_length_feature_set = pickle.load(fid)

        # with open(config.en_punctuation_feature_set_path, 'rb') as fid:
        #     en_punctuation_feature_set = pickle.load(fid)


    # Loading Spanish Features
    # if language <= 0:
    #     with open(config.sp_authors_path, 'rb') as fid:
    #         sp_authors = pickle.load(fid)

    #     with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
    #         sp_char_ngram_feature_set = pickle.load(fid)

    #     with open(config.sp_word_ngram_feature_set_path, 'rb') as fid:
    #         sp_word_ngram_feature_set = pickle.load(fid)

    #     with open(config.sp_avg_sentence_length_feature_set_path, 'rb') as fid:
    #         sp_avg_sentence_length_feature_set = pickle.load(fid)

    #     with open(config.sp_punctuation_feature_set_path, 'rb') as fid:
    #         sp_punctuation_feature_set = pickle.load(fid)

    # FEATURE NORMALIZATION AND NORMALIZATION #
    t0 = time.time()
    
    # English Features
    if language >= 0:
        current_features =\
        [(en_char_ngram_feature_set, '- Character N-Grams, Amount: ' + str(config.char_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (en_word_ngram_feature_set, '- Word N-Grams, Amount: ' + str(config.word_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (en_avg_sentence_length_feature_set, '- Average Sentence Length\n')]

        # Go through every combination of the features
        for j in range(len(current_features)):
            for subset in combinations(current_features, j + 1):

                en_training_feature_set = [] # final training feature set
                en_feature_text = ''.join([m[1] for m in subset]) # gets what features are being tested
                
                # Assembling them such that they look like x
                for i in range(len(en_authors)):
                    curr_feature = []
                    for k in range(len(subset)):
                        if subset[k][1][:3] == '- A':
                            curr_feature.append(subset[k][0][i])
                        else:
                            curr_feature.extend(subset[k][0][i])
                    en_training_feature_set.append(curr_feature)

                en_training_feature_set_normalised = normalize(en_training_feature_set, norm=config.normalization_type)
                test_classifier_kernels(en_authors, en_training_feature_set_normalised, en_feature_text, config.ENGLISH)

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
