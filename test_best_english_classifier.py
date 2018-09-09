from datetime import datetime
import config
import pickle
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

from test_classifier import cv_classifier_score

curr_time = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss-%fms")

lang = 'en'
lang_name = config.EN_NAME

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

current_features =\
[(en_char_ngram_feature_set, '- Character N-Grams, Amount: ' + str(config.char_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
(en_word_ngram_feature_set, '- Word N-Grams, Amount: ' + str(config.word_ngram_length) + ', ' + str(config.ngram_common_words) + '\n'),\
(en_avg_sentence_length_feature_set, '- Average Sentence Length\n'),\
(en_punctuation_feature_set, '- Punctuation Frequency\n')]

classifiers = [(LinearSVC(), 'Linear SVC'), (SVC(kernel='linear'), 'SVC with Linear Kernel'),\
    (SVC(kernel='poly'), 'SVC with Poly Kernel'), (SVC(kernel='rbf'), 'SVC with rbf Kernel'),\
    (SVC(kernel='sigmoid'), 'SVC with Sigmoid Kernel'), (GaussianNB(), 'Gaussian Naive Bayes'),\
    (BernoulliNB(), 'Bernoulli Naive Bayes'), (MultinomialNB(), 'Multinomial Naive Bayes'),\
    (KNeighborsClassifier(), 'KNeighbours'), (MLPClassifier(), 'Multi-Layer Perceptron'),\
    (SGDClassifier(max_iter=5, tol=None, shuffle=True), 'Stochastic Gradient Descent'),\
    (DecisionTreeClassifier(), 'Decision Trees'), (GaussianProcessClassifier(), 'Gaussian Process'),\
    (RandomForestClassifier(), 'Random Forest'), (AdaBoostClassifier(), 'AdaBoost'),\
    (QuadraticDiscriminantAnalysis(), 'Quadratic Discriminant Analysis')]

best_classifiers = []
file = open(config.en_best_classifier_text_path)
for line in file:
    best_classifiers.append([int(i) for i in line.split()])

output_file = open('./best_english_classifiers/' + lang + '_best_english_classifier_kernel_test_' + curr_time + '.txt', 'a+')

# Heading file and terminal output
print("TEST: " + lang_name + " Best Classifier Kernels " + curr_time + '\n')
output_file.write("TEST: " + lang_name + " Best Classifier Kernels " + curr_time + '\n\n')

for entry in best_classifiers:
    current_classifier = classifiers[entry[-1]]
    
    feature_set = []
    for i in range(len(entry) - 1):
        feature_set.append(current_features[entry[i]])

    en_training_feature_set = [] # final training feature set
    en_feature_text = ''.join([m[1] for m in feature_set]) # gets what features are being tested
    
    print(en_feature_text)
    # Assembling them such that they look like x
    for i in range(len(en_authors)):
        curr_feature = []
        for k in range(len(feature_set)):
            if feature_set[k][1][:3] == '- A':
                curr_feature.append(feature_set[k][0][i])
            else:
                curr_feature.extend(feature_set[k][0][i])
        en_training_feature_set.append(curr_feature)

    features_normalized = normalize(en_training_feature_set, norm=config.normalization_type)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_score = cv_classifier_score(current_classifier[0], features_normalized, en_authors, config.en_k_folds)
        output_file.write(en_feature_text)
        output_file.write(current_classifier[1] + ' ' + clf_score + '\n')
        output_file.write("\n")
        print(current_classifier[1], clf_score + "\n")

output_file.close()