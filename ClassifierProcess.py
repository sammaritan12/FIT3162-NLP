from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
from sklearn.preprocessing import normalize
import pickle
import config
from sys import argv
import time

def file_extraction():
    """
    Extracts files from specified folder and returns them as a list of strings
    [A, B, ... , C]
    
    Also returns authors, in the same order of texts
    [Author of A, Author of B, ... , Author C]
    """
    #### FILE EXTRACTION ####
    t0 = time.time()

    # lists all text files within the path provided
    file_names = list_filenames(config.en_processed_text_path)

    # gets filenames and converts to strings
    processed_texts = [filename_to_text(i) for i in file_names]

    # Extract authors
    authors = [i.splitlines()[0] for i in processed_texts]

    print("Text Import and Author Extraction Time:", time.time() - t0)

    return processed_texts, authors

def feature_extraction(processed_texts):
    """
    Extracts language features from processed texts
    - processed_texts is a list of strings [A, B, ... , C]

    Join features such that there consists the following list:
    [Text Features A, Text Features B, ... , Text Features C]
    Text Features: [Average Sentence Length, 
                    Punctuation 1 Freq, ... , Punctuation 14 Freq,
                    POS 1 Freq, ... , POS 8 Freq,
                    Most Common N-Gram Freq, ... , 1000th Most Common N-Gram Freq]
    
    Returns:
    - training_feature_set , which is list above
    - training_word_ngrams , word ngrams used to assemble word ngram frequencies in training feature set
    - training_char_ngrams , character ngrams used to assemble character ngram frequencies in training feature set
    """
    
    t0 = time.time()

    tokenized_words = [word_tokenize(i) for i in processed_texts]

    print("Word Tokenization Time:", time.time() - t0)
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
    avg_sentence_length_feature_set = [avg_sentence_length(processed_texts[i], tokenized_words[i]) for i in range(len(processed_texts))]

    print("Average Sentence Length Time:", time.time() - t0)
    t0 = time.time()

    training_feature_set = []

    # Assembling them such that they look like x
    for i in range(len(avg_sentence_length_feature_set)):
        training_feature_set.append([avg_sentence_length_feature_set[i]] + char_ngram_feature_set[i] + word_ngram_feature_set[i])
        
    
    print("Zipping Features Time:", time.time() - t0)
    t0 = time.time()

    return training_feature_set, training_word_ngrams, training_char_ngrams

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
    processed_texts, authors = file_extraction()

    #### FEATURE EXTRACTION AND ASSEMBLY ####
    training_feature_set, training_word_ngrams, training_char_ngrams = feature_extraction(processed_texts)

    #### FEATURE NORMALIZATION ####
    t0 = time.time()

    # Extract character ngrams from text
    ngram_text_dists = [FreqDist(char_ngram(config.ngram_length, word_tokenize(i))) for i in processed_texts]
    # collate ngram most common ngrams and their occurrences
    training_words, ngram_feature_set = ngram_selection(ngram_text_dists, config.ngram_common_words)
    # collate punctuation occurrences
    punctuation, punctuation_feature_set = punctuation_frequency(ngram_text_dists)
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

    print("Normalization Time:", time.time() - t0)
    t0 = time.time()

    #### CLASSIFIER TRAINING ####
    eng_classifier = english_classifier(training_feature_set_normalised, authors)

    print("Classifier Fit Time:", time.time() - t0)
    

    #### FILE SAVING ####
    t0 = time.time()

    # Save the classifier
    with open('eng_classifier.pkl', 'wb') as fid:
        pickle.dump(eng_classifier, fid)

    # Save training words used for character ngrams
    with open('eng_training_char_ngrams.pkl', 'wb') as fid:
        pickle.dump(training_char_ngrams, fid)

    # Save training ngrams used for word ngrams
    with open('eng_training_word_ngrams.pkl', 'wb') as fid:
        pickle.dump(training_word_ngrams, fid)
    
    print("File Save Time:", time.time() - t0)

    # Output success
    print('Classifier Successfully Created and Saved.')
