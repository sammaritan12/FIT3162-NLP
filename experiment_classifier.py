import pickle
import time
import warnings
from datetime import datetime
from io import TextIOWrapper
from itertools import combinations
from sys import argv

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import config
from classifier.english_classifier import english_classifier


def print_write(fid, text):
    '''
    Prints and writes to file at the same time
    Params:
    - fid, opened file to write to, has to have write access
    - text, string to print and write output
    '''
    if type(text) is not str:
        raise TypeError('text must be a str')
    if type(fid) is not TextIOWrapper:
        raise TypeError('fid must be a file')

    fid.write(text + '\n')
    print(text)


def cv_classifier_score(classifier, train, target, k_fold):
    '''
    Determines and outputs the accuracy of classifiers and their features using cross validation
    Params:
    - classifier, classifier object from sklearn
    - train, 2D array containing features from each author
    - target, list containing authors, index matched with train param
    - k_fold, positive integer used to determine k folds for cross validation
    '''
    # Type and Value checking
    if type(k_fold) is not int:
        raise TypeError("k_fold must be an integer")
    
    if k_fold <= 0:
        raise ValueError("k_fold must be a positive integer")

    # Run cross validation and return accuracy
    cv_score =  cross_val_score(classifier, train, target, cv=k_fold)
    return "Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2)


def test_classifier_kernels(authors, features_normalized, features_text, language, k_fold):
    """
    Creates a text file which outputs the results for english classifier kernel test.
    Params:
    - authors, the authors within the classifier, the target file
    - features_normalized, normalized feature set to be used by the classifier
    - features_text, text of what is included in the feature set being passed
    """
    # Type and Value checking
    if type(features_text) is not str:
        raise TypeError("features_text should be a string of features")

    if type(language) is not int:
        raise TypeError("language should an integer")

    if language not in [config.SPANISH, config.ENGLISH]:
        raise ValueError("language should either be config.SPANISH or config.ENGLISH")

    # Initialise values
    curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss-%fms")
    lang = 'unknown'
    lang_name = 'unknown'

    # Determine language to experiment
    if language == config.ENGLISH:
        lang = 'en'
        lang_name = config.EN_NAME
    elif language == config.SPANISH:
        lang = 'sp'
        lang_name = config.SP_NAME

    # Open file output for experiment results
    output_file = open(config.experiment_results_path + lang + '_classifier_kernel_test_' + curr_time + '.txt', 'a+')
    
    # Heading file and terminal output
    print_write(output_file, "TEST: " + lang_name + " Classifier Kernels " + curr_time + '\n')

    # Features file and terminal output
    print_write(output_file, "Features:\n" + features_text)

    # Normalization file and terminal output
    print_write(output_file, 'Using L2 Normalisation, Least Squares\n')

    # List of classifiers we are testing
    classifiers = [(LinearSVC(), 'Linear SVC'), (SVC(kernel='linear'), 'SVC with Linear Kernel'),\
        (SVC(kernel='poly'), 'SVC with Poly Kernel'), (SVC(kernel='rbf'), 'SVC with rbf Kernel'),\
        (SVC(kernel='sigmoid'), 'SVC with Sigmoid Kernel'), (GaussianNB(), 'Gaussian Naive Bayes'),\
        (BernoulliNB(), 'Bernoulli Naive Bayes'), (MultinomialNB(), 'Multinomial Naive Bayes'),\
        (KNeighborsClassifier(), 'KNeighbours'), (MLPClassifier(), 'Multi-Layer Perceptron'),\
        (SGDClassifier(max_iter=5, tol=None, shuffle=True), 'Stochastic Gradient Descent'),\
        (DecisionTreeClassifier(), 'Decision Trees'), (GaussianProcessClassifier(), 'Gaussian Process'),\
        (RandomForestClassifier(), 'Random Forest'), (AdaBoostClassifier(), 'AdaBoost'),\
        (QuadraticDiscriminantAnalysis(), 'Quadratic Discriminant Analysis')]

    # Testing the classifier accuracy
    for clf, text in classifiers:
        clf_score = cv_classifier_score(clf, features_normalized, authors, k_fold)
        print_write(output_file, text + ' ' + clf_score)

    print('')

    output_file.close()


if __name__ == '__main__':
    '''
    File used to experiment all possible classifiers, whether English or Spanish
    File can be run as:
    python experiment_classifier.py <english/spanish>
    If no language is specified it is assumed to be both
    '''
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
        # Tuples for feature set and description, used for print output
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
                
                # Assembling them such that they look like x, basically transposing a matrix
                for i in range(len(en_authors)):
                    curr_feature = []
                    for k in range(len(subset)):
                        if subset[k][1] == '- Average Sentence Length\n':
                            curr_feature.append(subset[k][0][i])
                        else:
                            curr_feature.extend(subset[k][0][i])
                    en_training_feature_set.append(curr_feature)
                    
                # Normalise the feature set
                en_training_feature_set_normalised = normalize(en_training_feature_set, norm=config.normalization_type)

                # Test the classifiers, ignores warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_classifier_kernels(en_authors, en_training_feature_set_normalised, en_feature_text, config.ENGLISH, config.en_k_folds)

    # Spanish Features
    if language <= 0:
        # Tuples for feature set and description, used for print output
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
                
                # Assembling them such that they look like x, basically transposing a matrix
                for i in range(len(sp_authors)):
                    curr_feature = []
                    for k in range(len(subset)):
                        if subset[k][1] == '- Average Sentence Length\n':
                            curr_feature.append(subset[k][0][i])
                        else:
                            curr_feature.extend(subset[k][0][i])
                    sp_training_feature_set.append(curr_feature)

                # Normalise the feature set
                sp_training_feature_set_normalised = normalize(sp_training_feature_set, norm=config.normalization_type)

                # Test the classifiers, ignores warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_classifier_kernels(sp_authors, sp_training_feature_set_normalised, sp_feature_text, config.SPANISH, config.sp_k_folds)

    print("Zipping and Normalisation Time:", time.time() - t0)
    t0 = time.time()
