#  THIS IS A TEST FOR ENGLISH CLASSIFIER

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# TODO vectoriser for texts, turns text to numbers
# this one vectorises 1grams
def textVectorizer(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

# BELOW IS A PLAN FOR HOW TO CREATE CLASSIFIERS
'''
1. For each author there is an SVM classifier
2. We use LinearSVC which is
    - multiclass classifier
    - uses one vs all instead of one vs one which is n(n-1)/2
3. The feature sets will be 
    a. preprocessed (nltk)
    b. vectorised (sklearn)
    c. normalised (sklearn)
4. Training data is then 'fitted' to the classifier with the answers
5. Test data is then predicted to the classifier
'''

def englishClassifier(vectorisedNormalisedData, dataAnswers):
    # vectorisedNormalisedData is a 2D list represented as follows
    # [[A], [B], ... , [C]]
    # Where A, B, C are training data for documents A, B, C
    # A, B, C are represented as:
    # [x, y, ..., z]
    # Where x, y, z are features of document A, B, C
    # dataAnswers is represented as follows:
    # [a, b, ..., c]
    # Where a, b, c are the authors of documents A, B, C and are matched via the index

    # LinearSVC is used as a multiclass, using a one vs all approach
    svm = LinearSVC()
    # Fitting training data with answers
    svm.fit(vectorisedNormalisedData, dataAnswers)
    return svm

def classifierPredict(classifier, vectorisedNormalisedTestData):
    # Predicts the test data from the items in the classifier, and returns ML guess
    return classifier.predict(vectorisedNormalisedTestData)
