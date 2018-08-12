from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
import pickle

# MAIN RUN FILE
# This is the main run file for the project, will be command line for now
if __name__ == "__main__":
    # First argument will be text file
    # Second argument is language, if empty use English

    # TODO process text file to string

    # TODO process text file to feature set

    # TODO normalise feature set

    # TODO load classifer
    with open('eng_classifier.pkl', 'rb') as fid:
        eng_classifier = pickle.load(fid)

    # TODO predict and output

    pass