from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
from sklearn.preprocessing import normalize
import pickle
import config
from sys import argv

if __name__ == "__main__":
    # THIS FILE WILL CREATE AND FIT THE CLASSIFIER, THEN SAVE CLASSIFIER
    # First argument specifies whether to build English, Spanish or Both
    # If empty, build both

    # # UNCOMMENT WHEN READY TO USE ARGUMENTS FOR ENGLISH AND SPANISH
    # language = 'both'

    # if len(argv) > 1:
    #     if argv[1].lower() == 'spanish':
    #         language = 'spanish'
    #     elif argv[1].lower() == 'english':
    #         language = 'english'
    #     elif argv[1].lower() == 'both':
    #         pass
    #     else:
    #         print("Please choose a language as argument, either 'english' 'spanish' or 'both' ")

    # Import processed gutenberg texts from folder and place as a list [A, B, ... , C]

    # lists all text files within the path provided
    file_names = list_filenames(config.processed_text_path)
    # gets filenames and converts to strings
    processed_texts = [filename_to_text(i) for i in file_names]

    # TODO Extract feature set and authors from gutenberg texts

    # Extract authors
    authors = [i.splitlines()[0] for i in processed_texts]

    # Extract character ngrams from text
    ngram_text_dists = [FreqDist(char_ngram(config.ngram_length, word_tokenize(i))) for i in processed_texts]
    # collate ngram most common ngrams and their ocurrences
    training_words, ngram_feature_set = ngram_selection(ngram_text_dists, config.ngram_common_words)

    # TODO Assemble feature set from gutenberg texts
    ### PUTTING IT ALL TOGETHER ###
    # Join features such that there consists 2 lists, x, y
    # x: [Text Features A, Text Features B, ... , Text Features C]
    # Text Features: [Average Sentence Length, 
    #                 Punctuation 1 Freq, ... , Punctuation 14 Freq,
    #                 POS 1 Freq, ... , POS 8 Freq,
    #                 Most Common N-Gram Freq, ... , 1000th Most Common N-Gram Freq]
    # y: [Author A, Author B, ... , Author C]

    training_feature_set = ngram_feature_set

    # TODO Normalise feature sets
    # L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
    # L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration
    training_feature_set_normalised = normalize(training_feature_set, norm=config.normalization_type)

    # Fit classifier
    eng_classifier = englishClassifier(training_feature_set_normalised, authors)

    # Save the classifier
    with open('eng_classifier.pkl', 'wb') as fid:
        pickle.dump(eng_classifier, fid)

    # Save training words used for ngrams
    with open('eng_training_words.pkl', 'wb') as fid:
        pickle.dump(training_words, fid)
    
    # Output success
    print('Classifier Successfully Created and Saved.')
