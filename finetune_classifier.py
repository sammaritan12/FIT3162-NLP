import pickle
from datetime import datetime
from sys import argv

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

import config
from experiment_classifier import cv_classifier_score, print_write


def finetune(lang, authors, char_ngrams, word_ngrams):
    '''
    Checks possible parameters for RandomForestClassifier and LinearSVC for language specified
    Params:
    - lang, either config.SPANISH or config.ENGLISH
    - authors, list of authors
    - char_ngrams, character n-grams feature set extracted
    - word_ngrams, word n-grams feature set extracted
    '''
    # Type checking
    if type(lang) is not int:
        raise TypeError("lang should be a string corresponding from config.py")

    # Initialise variables
    lang_name = 'Unknown'
    lang_name_short = 'uk'
    feature_combo = list(map(lambda x: normalize(x,norm=config.normalization_type), \
        [char_ngrams, word_ngrams, [char_ngrams[i] + word_ngrams[i] for i in range(len(char_ngrams))]]))
    feature_combo_text = ['Character Ngrams', 'Word Ngrams', 'Character Ngrams, Word Ngrams']
    k_folds = 1

    # Language checking
    if lang == config.SPANISH:
        lang_name = config.SP_NAME
        lang_name_short = 'sp'
        k_folds = config.sp_k_folds
    elif lang == config.ENGLISH:
        lang_name = config.EN_NAME
        lang_name_short = 'en'
        k_folds = config.en_k_folds
    else:
        raise ValueError("language should either be English or Spanish")

    # Initialise output file
    curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss-%fms")
    output_file = open(config.experiment_results_path + lang_name_short + '_finetine_kernel_test_' + curr_time + '.txt', 'a+')

    print_write(output_file, 'Test: ' + lang_name + ' Finetuning Kernels' + curr_time)

    # Test LinearSVC and RandomForestClassifier
    for f in range(len(feature_combo)):        
        print_write(output_file, 'Feature/s: ' + feature_combo_text[f])

        test_linearsvc(feature_combo[f], authors, output_file, k_folds)
        test_randomforest(feature_combo[f], authors, output_file, k_folds)

    output_file.close()

def test_linearsvc(train, target, output_file, k_folds):
    '''
    Tests various parameters and their effectiveness and accuracy for RandomForestClassifier
    Params:
    - train, 2D array of features aligned with target list
    - target, list of features that correspond with train param
    - output_file, opened file with write permission
    - k_folds, int to test how many k folds for cross validation
    '''
    print_write(output_file, 'Classifier: LinearSVC')

    # Default, no params specified
    clf = LinearSVC()
    print_write(output_file, 'Default')
    output = cv_classifier_score(clf, train, target, k_folds)
    print_write(output_file, output + '\n')

    # Possible Params
    params = {
        'max_iter' : [1500, 2000, 3000],
        'multi_class' : ['crammer_singer'],
        'C' : [0.25, 0.5, 2, 5, 10, 20],
        'loss' : ['hinge']

    }
    
    # For every parameter, run cross validation testing and output the accuracy
    for field in params:
        for test_val in params[field]:
            clf = RandomForestClassifier()
            print_write(output_file, field + ': ' + str(test_val))
            output = cv_classifier_score(clf, train, target, k_folds)
            print_write(output_file, output + '\n')


def test_randomforest(train, target, output_file, k_folds):
    '''
    Tests various parameters and their effectiveness and accuracy for RandomForestClassifier
    Params:
    - train, 2D array of features aligned with target list
    - target, list of features that correspond with train param
    - output_file, opened file with write permission
    - k_folds, int to test how many k folds for cross validation
    '''
    print_write(output_file, 'Classifier: RandomForest')
    
    # Default, no params specified
    clf = RandomForestClassifier()
    print_write(output_file, 'Default')
    output = cv_classifier_score(clf, train, target, k_folds)
    print_write(output_file, output + '\n')

    # Possible Params
    params = {
        'n_estimators' : [5,20,50,100,200],
        'max_features': [None, 'log2'],
        'min_samples_leaf' : [2, 10, 25, 50],
        'min_samples_split' : [3, 5, 10, 25]}

    # For every parameter, run cross validation testing and output the accuracy
    for field in params:
        for test_val in params[field]:
            clf = RandomForestClassifier(**{field: test_val})
            print_write(output_file, field + ': ' + str(test_val))
            output = cv_classifier_score(clf, train, target, k_folds)
            print_write(output_file, output + '\n')

    # Testing n_estimators=100 and max_features=None
    clf = RandomForestClassifier(n_estimators=100, max_features=None)
    print_write(output_file, 'N Estimators: 100, Max Features: None')
    output = cv_classifier_score(clf, train, target, k_folds)
    print_write(output_file, output + '\n')

    # Testing min_samples_split=10 and n_estimators=100
    clf = RandomForestClassifier(min_samples_split=10, n_estimators=100)
    print_write(output_file, 'Min Sample Split: 10, N Estimators: 100')
    output = cv_classifier_score(clf, train, target, k_folds)
    print_write(output_file, output + '\n')

if __name__ == '__main__':
    '''
    Finetunes RandomForestClassifier and LinearSVC, run as:
    python finetune_classifier.py <english/spanish>
    If language is not specified, both are assumed to be tested
    '''
    language = 0

    # Checking language
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


    # Loading Spanish Features
    if language <= 0:
        with open(config.sp_authors_path, 'rb') as fid:
            sp_authors = pickle.load(fid)

        with open(config.sp_char_ngram_feature_set_path, 'rb') as fid:
            sp_char_ngram_feature_set = pickle.load(fid)

        with open(config.sp_word_ngram_feature_set_path, 'rb') as fid:
            sp_word_ngram_feature_set = pickle.load(fid)

    # English Testing
    if language >= 0:
        finetune(config.ENGLISH, en_authors, en_char_ngram_feature_set, en_word_ngram_feature_set)

    # Spanish Testing
    if language <= 0:
        finetune(config.SPANISH, sp_authors, sp_char_ngram_feature_set, sp_word_ngram_feature_set)
