import pickle
from sys import argv

from nltk import word_tokenize
from sklearn.preprocessing import normalize

import config
from extraction.feature_extraction import (avg_sentence_length, char_ngram,
                                           freqdist_selection, word_ngram)
from extraction.file_extraction import filename_to_text, list_filenames

# MAIN RUN FILE
# This is the main run file for the project, will be command line for now
if __name__ == "__main__":

    # UNCOMMENT WHEN READY TO USE SYSTEM ARGUMENTS
    english = True
    
    # First argument will be text file, else quit
    if len(argv) < 2:
        print("Please enter a gutenberg file to analyse")
        quit()

    # Second argument is language, if empty use english
    if len(argv) > 2 and argv[2].lower()[:2] == 'sp':
        english = False
    elif len(argv) > 2 and argv[2].lower()[:2] == 'en':
        pass
    else:
        print("Please choose a valid language, spanish or english")

    # process text file to string
    text = filename_to_text(argv[1])

    if english:
        # loads classifier
        with open(config.en_classifier_path, 'rb') as fid:
            classifier = pickle.load(fid)

        # loads words used in ngrams
        with open(config.en_training_char_ngrams_path, 'rb') as fid:
            training_char_ngrams = pickle.load(fid)

        # loads words used in ngrams
        with open(config.en_training_word_ngrams_path, 'rb') as fid:
            training_word_ngrams = pickle.load(fid)
    else:
        with open(config.sp_classifier_path, 'rb') as fid:
            classifier = pickle.load(fid)

        # loads words used in ngrams
        with open(config.sp_training_char_ngrams_path, 'rb') as fid:
            training_char_ngrams = pickle.load(fid)

        # loads words used in ngrams
        with open(config.sp_training_word_ngrams_path, 'rb') as fid:
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

    # Character ngrams
    char_ngrams_feature_set = \
        freqdist_test_selection(FreqDist(char_ngram(config.char_ngram_length, tokenized_word)), training_char_ngrams)

    # Word ngrams
    word_ngrams_feature_set = \
        freqdist_test_selection(FreqDist(word_ngram(config.word_ngram_length, tokenized_word)), training_word_ngrams)

    # Average sentence length
    avg_sentence_length_feature_set = avg_sentence_length(text, tokenized_word)

    # Aggregated feature set
    test_feature_set = char_ngrams_feature_set + word_ngrams_feature_set + [avg_sentence_length_feature_set]

    # Normalise feature set
    test_feature_set_normalised = normalize([test_feature_set], norm=config.normalization_type)

    # Predict and output
    prediction = classifier.predict(test_feature_set_normalised)
    print('Predicted Author:', prediction[0])
