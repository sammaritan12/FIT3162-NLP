from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
# BELOW IS A PLAN FOR HOW TO CREATE CLASSIFIERS

# 1. For each author there is an SVM classifier
# 2. We use LinearSVC which is
#     - multiclass classifier
#     - uses one vs all instead of one vs one which is n(n-1)/2
# 3. The feature sets will be 
#     a. preprocessed (nltk)
#     b. vectorised (sklearn)
#     c. normalised (sklearn)
# 4. Training data is then 'fitted' to the classifier with the answers
# 5. Test data is then predicted to the classifier


### VECTORIZER FEATURES ###
# [x] Turn text to numbers

def text_vectorizer(texts):
    """
    Turns texts to numbers
    - texts is a list of strings
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

### NORMALISATION ###


### CLASSIFICATION ###
# [x] english classifier
# [] Tweak english classifier for highest accuracy
# [] spanish classifier
# [] Tweak spanish classifier for highest accuracy

def english_classifier(vectorised_normalised_data, data_answers):
    """
    Fits normalised data to LinearSVC

    vectorisedNormalisedData is a 2D list represented as follows
    [[A], [B], ... , [C]]
    Where A, B, C are training data for documents A, B, C
    A, B, C are represented as:
    [x, y, ..., z]
    Where x, y, z are features of document A, B, C
    dataAnswers is represented as follows:
    [a, b, ..., c]
    Where a, b, c are the authors of documents A, B, C and are matched via the index
    """
    # LinearSVC is used as a multiclass, using a one vs all approach
    svm = SVC(kernel="rbf", verbose=True)

    # Fitting training data with answers
    svm.fit(vectorised_normalised_data, data_answers)
    return svm
