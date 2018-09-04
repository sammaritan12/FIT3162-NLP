import pickle
import time
from datetime import datetime
from itertools import combinations
from sys import argv
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import config
from classifier.english_classifier import english_classifier


def cv_classifier_score(classifier, train, target, k_fold):
    cv_score =  cross_val_score(classifier, train, target, cv=k_fold)
    return "Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2)


def test_classifier_kernels(authors, features_normalized, features_text, language, k_fold):
    """
    Creates a text file which outputs the results for english classifier kernel test.
    - authors, the authors within the classifier, the target file
    - features_normalized, normalized feature set to be used by the classifier
    - features_text, text of what is included in the feature set being passed
    """
    curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss-%fms")
    lang = 'unknown'
    lang_name = 'unknown'

    if language == config.ENGLISH:
        lang = 'en'
        lang_name = config.EN_NAME
    elif language == config.SPANISH:
        lang = 'sp'
        lang_name = config.SP_NAME

    output_file = open('./testing/' + lang + '_classifier_kernel_test_' + curr_time + '.txt', 'a+')

    
    # Heading file and terminal output
    print("TEST: " + lang_name + " Classifier Kernels " + curr_time + '\n')
    output_file.write("TEST: " + lang_name + " Classifier Kernels " + curr_time + '\n\n')

    # Features file and terminal output
    print("Features:\n" + features_text)
    output_file.write("Features:\n" + features_text +"\n")

    # Normalization file and terminal output
    output_file.write('Using L2 Normalisation, Least Squares\n\n')
    print('Using', config.normalization_type.upper(), 'Normalisation\n')

    classifiers = [(LinearSVC(), 'Linear SVC'), (SVC(kernel='linear'), 'SVC with Linear Kernel'),\
        (SVC(kernel='poly'), 'SVC with Poly Kernel'), (SVC(kernel='rbf'), 'SVC with rbf Kernel'),\
        (SVC(kernel='sigmoid'), 'SVC with Sigmoid Kernel'), (GaussianNB(), 'Gaussian Naive Bayes'),\
        (BernoulliNB(), 'Bernoulli Naive Bayes'), (MultinomialNB(), 'Multinomial Naive Bayes'),\
        (KNeighborsClassifier(), 'KNeighbours'), (MLPClassifier(), 'Multi-Layer Perceptron'),\
        (SGDClassifier(max_iter=5, tol=None, shuffle=True), 'Stochastic Gradient Descent'),\
        (DecisionTreeClassifier(), 'Decision Trees'), (GaussianProcessClassifier(), 'Gaussian Process'),\
        (RandomForestClassifier(), 'Random Forest'), (AdaBoostClassifier(), 'AdaBoost'),\
        (QuadraticDiscriminantAnalysis(), 'Quadratic Discriminant Analysis')]

    for clf, text in classifiers:
        clf_score = cv_classifier_score(clf, features_normalized, authors, k_fold)
        output_file.write(text + ' ' + clf_score + '\n')
        print(text, clf_score)

    print('')

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

        with open(config.en_punctuation_feature_set_path, 'rb') as fid:
            en_punctuation_feature_set = pickle.load(fid)


    # Loading Spanish Features
    if language <= 0:
        with open(config.sp_authors_path, 'rb') as fid:
            sp_authors = pickle.load(fid)

        with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
            sp_char_ngram_feature_set = pickle.load(fid)

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
        current_features =\
        [(en_char_ngram_feature_set, '- Character N-Grams, Amount: ' + str(config.char_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (en_word_ngram_feature_set, '- Word N-Grams, Amount: ' + str(config.word_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (en_avg_sentence_length_feature_set, '- Average Sentence Length\n'),\
        (en_punctuation_feature_set, '- Punctuation Frequency\n')]

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

                # Test the classifiers, ignores warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_classifier_kernels(en_authors, en_training_feature_set_normalised, en_feature_text, config.ENGLISH, config.en_k_folds)

    # Spanish Features
    if language <= 0:
        current_features =\
        [(sp_char_ngram_feature_set, '- Character N-Grams, Amount: ' + str(config.char_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (sp_word_ngram_feature_set, '- Word N-Grams, Amount: ' + str(config.word_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
        (sp_avg_sentence_length_feature_set, '- Average Sentence Length\n'),\
        (sp_punctuation_feature_set, '- Punctuation Frequency')]

        # Go through every combination of the features
        for j in range(len(current_features)):
            for subset in combinations(current_features, j + 1):

                sp_training_feature_set = [] # final training feature set
                sp_feature_text = ''.join([m[1] for m in subset]) # gets what features are being tested
                
                # Assembling them such that they look like x
                for i in range(len(sp_authors)):
                    curr_feature = []
                    for k in range(len(subset)):
                        if subset[k][1][:3] == '- A':
                            curr_feature.append(subset[k][0][i])
                        else:
                            curr_feature.extend(subset[k][0][i])
                    sp_training_feature_set.append(curr_feature)

                sp_training_feature_set_normalised = normalize(sp_training_feature_set, norm=config.normalization_type)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_classifier_kernels(sp_authors, sp_training_feature_set_normalised, sp_feature_text, config.SPANISH, config.sp_k_folds)

    print("Zipping and Normalisation Time:", time.time() - t0)
    t0 = time.time()
