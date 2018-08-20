from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
from sklearn.preprocessing import normalize
import pickle
import config
from sys import argv
import time

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

    #### FILE EXTRACTION ####
    # Import processed gutenberg texts from folder and place as a list [A, B, ... , C]

    # lists all text files within the path provided
    file_names = list_filenames(config.processed_text_path)
    # gets filenames and converts to strings
    processed_texts = [filename_to_text(i) for i in file_names]

    # Extract authors
    authors = [i.splitlines()[0] for i in processed_texts]

    #### FEATURE EXTRACTION ####
    tokenized_words = [word_tokenize(i) for i in processed_texts]

    t0 = time.time()
    # Extract character ngrams from text
    char_ngram_text_dists = [FreqDist(char_ngram(config.char_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]
    
    # collate character ngram most common ngrams and their ocurrences
    training_char_ngrams, char_ngram_feature_set = freqdist_selection(char_ngram_text_dists, config.ngram_common_words)
    
    print("Character Ngrams Time:", time.time() - t0)
    t0 = time.time()

    # Extract word ngrams from text
    word_ngram_text_dists = [FreqDist(word_ngram(config.word_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]

    # Collate word ngram most common ngrams and their occurrences
    training_word_ngrams, word_ngram_feature_set = freqdist_selection(word_ngram_text_dists, config.ngram_common_words)

    print("Word Ngrams Time:", time.time() - t0)
    t0 = time.time()

    # Average sentence length
    avg_sentence_length_feature_set = [avg_sentence_length(i) for i in processed_texts]

    print("Average Sentence Length Time:", time.time() - t0)
    t0 = time.time()

    # TODO Assemble feature set from gutenberg texts
    ### PUTTING IT ALL TOGETHER ###
    # Join features such that there consists 2 lists, x, y
    # x: [Text Features A, Text Features B, ... , Text Features C]
    # Text Features: [Average Sentence Length, 
    #                 Punctuation 1 Freq, ... , Punctuation 14 Freq,
    #                 POS 1 Freq, ... , POS 8 Freq,
    #                 Most Common N-Gram Freq, ... , 1000th Most Common N-Gram Freq]
    # y: [Author A, Author B, ... , Author C]

    training_feature_set = []

    # Assembling them such that they look like x
    for i in range(len(avg_sentence_length_feature_set)):
        training_feature_set.append([avg_sentence_length_feature_set[i]] + char_ngram_feature_set[i] + word_ngram_feature_set[i])
        
    
    print("Zipping Features Time:", time.time() - t0)
    t0 = time.time()

    print(training_feature_set[0])

    # TODO Normalise feature sets
    # L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
    # L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration
    training_feature_set_normalised = normalize(training_feature_set, norm=config.normalization_type)

    # Fit classifier
    eng_classifier = englishClassifier(training_feature_set_normalised, authors)

    #### FILE SAVING ####

    # Save the classifier
    with open('eng_classifier.pkl', 'wb') as fid:
        pickle.dump(eng_classifier, fid)

    # Save training words used for character ngrams
    with open('eng_training_char_ngrams.pkl', 'wb') as fid:
        pickle.dump(training_char_ngrams, fid)

    # Save training ngrams used for word ngrams
    with open('eng_training_word_ngrams.pkl', 'wb') as fid:
        pickle.dump(training_word_ngrams, fid)
    
    # Output success
    print('Classifier Successfully Created and Saved.')
