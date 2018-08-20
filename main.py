from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
from sklearn.preprocessing import normalize
import pickle
import config
from sys import argv

# MAIN RUN FILE
# This is the main run file for the project, will be command line for now
if __name__ == "__main__":

    # UNCOMMENT WHEN READY TO USE SYSTEM ARGUMENTS
    english = True
    
    # First argument will be text file, else quit
    if len(argv) < 2:
        print("Please enter a gutenberg file to analyse")
        quit()
    
    # Second argument is language, if empty use English
    if len(argv) > 3 and argv[2].lower() == 'spanish':
        language = False

    # process text file to string
    text = filename_to_text(argv[1])

    # loads classifer
    with open('eng_classifier.pkl', 'rb') as fid:
        eng_classifier = pickle.load(fid)

    # loads words used in ngrams
    with open('eng_training_char_ngrams.pkl', 'rb') as fid:
        training_char_ngrams = pickle.load(fid)

    # loads words used in ngrams
    with open('eng_training_word_ngrams.pkl', 'rb') as fid:
        training_word_ngrams = pickle.load(fid)

    # # Exactly the same as Text A in training data
    # text = '''There were a king with a large jaw and a queen with a plain face, on the
    # throne of England; there were a king with a large jaw and a queen with
    # a fair face, on the throne of France. In both countries it was clearer
    # than crystal to the lords of the State preserves of loaves and fishes,
    # that things in general were settled for ever.'''

    # TODO process text file to feature set

    # ngram feature set extraction
    tokenized_word = word_tokenize(text)

    # character ngrams
    char_ngrams_feature_set = freqdist_test_selection(FreqDist(char_ngram(config.char_ngram_length, tokenized_word)), training_char_ngrams)

    # word ngrams
    word_ngrams_feature_set = freqdist_test_selection(FreqDist(word_ngram(config.word_ngram_length, tokenized_word)), training_word_ngrams)

    # Average sentence length
    avg_sentence_length_feature_set = avg_sentence_length(text)

    # Aggregated feature set
    test_feature_set = char_ngrams_feature_set + word_ngrams_feature_set + [avg_sentence_length_feature_set]

    

    # normalise feature set
    # L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
    # L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration
    # Must be the same from ClassifierProcess.py
    test_feature_set_normalised = normalize([test_feature_set], norm=config.normalization_type)

    # predict and output
    prediction = eng_classifier.predict(test_feature_set_normalised)
    print('Predicted Author:', prediction[0])