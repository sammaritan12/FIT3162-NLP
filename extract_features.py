from extraction.feature_extraction import freqdist_selection, char_ngram, word_ngram, avg_sentence_length
from extraction.file_extraction import filename_to_text, list_filenames
import pickle
import config
from sys import argv
import time
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


def file_extraction():
    """
    Extracts files from specified folder and returns them as a list of strings
    [A, B, ... , C]
    
    Also returns authors, in the same order of texts
    [Author of A, Author of B, ... , Author C]
    """
    # FILE EXTRACTION #
    curr_time = time.time()

    # lists all text files within the path provided
    file_names = list_filenames(config.en_processed_text_path)

    # gets file names and converts to strings
    processed_texts = [filename_to_text(i) for i in file_names]

    # Extract authors
    authors = [i.splitlines()[0] for i in processed_texts]

    print("Text Import and Author extraction Time:", time.time() - curr_time)

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
    
    curr_time = time.time()

    tokenized_words = [word_tokenize(i) for i in processed_texts]

    print("Word Tokenization Time:", time.time() - curr_time)
    curr_time = time.time()
    
    # Extract character ngrams from text
    char_ngram_text_dists =\
        [FreqDist(char_ngram(config.char_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]
    
    # collate character ngram most common ngrams and their occurrences
    training_char_ngrams, char_ngram_feature_set = freqdist_selection(char_ngram_text_dists, config.ngram_common_words)
    
    print("Character Ngrams Time:", time.time() - curr_time)
    curr_time = time.time()

    # Extract word ngrams from text
    word_ngram_text_dists =\
        [FreqDist(word_ngram(config.word_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]

    # Collate word ngram most common ngrams and their occurrences
    training_word_ngrams, word_ngram_feature_set =\
        freqdist_selection(word_ngram_text_dists, config.ngram_common_words)

    print("Word Ngrams Time:", time.time() - curr_time)
    curr_time = time.time()

    # Average sentence length
    avg_sentence_length_feature_set =\
        [avg_sentence_length(processed_texts[i], tokenized_words[i]) for i in range(len(processed_texts))]

    print("Average Sentence Length Time:", time.time() - curr_time)

    return char_ngram_feature_set, training_char_ngrams, \
        word_ngram_feature_set, training_word_ngrams, \
        avg_sentence_length_feature_set


if __name__ == "__main__":
    # THIS FILE WILL IMPORT THE TEXTS AND EXTRACT THE FEATURES, THEN SAVE THEM AS PICKLES
    # First argument specifies whether to extract english, spanish or Both
    # If empty, extract both

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

    # FILE EXTRACTION #
    # Import processed gutenberg texts from folder and place as a list [A, B, ... , C]
    en_processed_texts, en_authors = file_extraction()

    # FEATURE EXTRACTION AND ASSEMBLY #
    en_char_ngram_feature_set, en_training_char_ngrams, \
        en_word_ngram_feature_set, en_training_word_ngrams, \
        en_avg_sentence_length_feature_set = feature_extraction(en_processed_texts)

    # FILE SAVING #
    t0 = time.time()

    # Save english training words used for character ngrams
    with open(config.en_training_char_ngrams_path, 'wb') as fid:
        pickle.dump(en_training_char_ngrams, fid)

    # Save english training ngrams used for word ngrams
    with open(config.en_training_word_ngrams_path, 'wb') as fid:
        pickle.dump(en_training_word_ngrams, fid)

    # Save english char ngram feature set
    with open(config.en_char_ngram_feature_set_path, 'wb') as fid:
        pickle.dump(en_char_ngram_feature_set, fid)

    # Save english word ngram feature set
    with open(config.en_word_ngram_feature_set_path, 'wb') as fid:
        pickle.dump(en_word_ngram_feature_set, fid)

    # save english average sentence length feature set
    with open(config.en_avg_sentence_length_feature_set_path, 'wb') as fid:
        pickle.dump(en_avg_sentence_length_feature_set, fid)
    
    print("File Save Time:", time.time() - t0)

    # Output success
    print('Features successfully extracted and saved.')
