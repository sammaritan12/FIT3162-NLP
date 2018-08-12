from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
import pickle

if __name__ == "__main__":
    # THIS FILE WILL CREATE AND FIT THE CLASSIFIER, THEN SAVE CLASSIFIER
    # First argument specifies whether to build English, Spanish or Both
    # If empty, build both
    pass

    # TODO Import raw gutenberg texts

    # TODO Preprocess raw gutenberg texts

    # TODO Extract feature set and authors from gutenberg texts

    # TODO Assemble feature set from gutenberg texts

    # TODO Normalise feature sets

    # TODO Fit classifier
    eng_classifier = englishClassifier(eng_train, eng_answer)

    # TODO Save the classifier
    with open('eng_classifier.pkl', 'wb') as fid:
        pickle.dump(eng_classifier, fid)
    
    # TODO Output success