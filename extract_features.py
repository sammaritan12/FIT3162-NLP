import pickle
import time
from sys import argv

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

import config
from extraction.feature_extraction import (avg_sentence_length, char_ngram,
                                           freqdist_selection,
                                           punctuation_frequency, word_ngram)
from extraction.file_extraction import filename_to_text, list_filenames


def file_extraction(language):
    """
    Extracts files from specified folder and returns them as a list of strings
    [A, B, ... , C]
    
    Also returns authors, in the same order of texts
    [Author of A, Author of B, ... , Author C]
    """
    if language != config.ENGLISH or language != config.SPANISH:
        raise ValueError("language must either be config.ENGLISH or config.SPANISH")

    # FILE EXTRACTION #
    curr_time = time.time()

    # lists all text files within the path provided
    lang_name = 'Unknown'
    # English
    if language == config.ENGLISH:
        file_names = list_filenames(config.en_processed_text_path)
        lang_name = config.EN_NAME
    # Spanish
    elif language == config.SPANISH:
        file_names = list_filenames(config.sp_processed_text_path)
        lang_name = config.SP_NAME

    # gets file names and converts to strings
    processed_texts = [filename_to_text(i) for i in file_names]

    # Extract authors
    authors = [i.splitlines()[0] for i in processed_texts]

    print(lang_name, "Text Import and Author extraction Time:", time.time() - curr_time)

    return processed_texts, authors


def feature_extraction(processed_texts, language):
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
    if language != config.ENGLISH and language != config.SPANISH:
        raise ValueError("language must either be config.ENGLISH or config.SPANISH")

    if type(processed_texts) is not list:
        raise TypeError("processed_texts must be a list of strings")

    if not all(isinstance(x, str) for x in processed_texts):
        raise ValueError("words must be a list of strings")

    if language == config.ENGLISH:
        lang_name = config.EN_NAME
    elif language == config.SPANISH:
        lang_name = config.SP_NAME
    else:
        lang_name = "Unknown"

    curr_time = time.time()

    tokenized_words = [word_tokenize(i) for i in processed_texts]

    print(lang_name, "Word Tokenization Time:", time.time() - curr_time)
    curr_time = time.time()
    
    # Extract character ngrams from text
    char_ngram_text_dists =\
        [FreqDist(char_ngram(config.char_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]
    
    # collate character ngram most common ngrams and their occurrences
    training_char_ngrams, char_ngram_feature_set = freqdist_selection(char_ngram_text_dists, config.ngram_common_words)
    
    print(lang_name, "Character Ngrams Time:", time.time() - curr_time)
    curr_time = time.time()

    # Extract word ngrams from text
    word_ngram_text_dists =\
        [FreqDist(word_ngram(config.word_ngram_length, tokenized_words[i])) for i in range(len(processed_texts))]

    # Collate word ngram most common ngrams and their occurrences
    training_word_ngrams, word_ngram_feature_set =\
        freqdist_selection(word_ngram_text_dists, config.ngram_common_words)

    print(lang_name, "Word Ngrams Time:", time.time() - curr_time)
    curr_time = time.time()

    # Average sentence length
    avg_sentence_length_feature_set =\
        [avg_sentence_length(processed_texts[i], tokenized_words[i]) for i in range(len(processed_texts))]

    print(lang_name, "Average Sentence Length Time:", time.time() - curr_time)
    curr_time = time.time()

    # Punctuation frequency
    punctuation_text_dists = [FreqDist(punctuation_frequency(tokenized_words[i])) for i in range(len(processed_texts))]

    training_punctuation, punctuation_feature_set = freqdist_selection(punctuation_text_dists, config.punctuation_length)

    print(lang_name, "Punctuation Distribution Length Time:", time.time() - curr_time)

    return char_ngram_feature_set, training_char_ngrams, \
        word_ngram_feature_set, training_word_ngrams, \
        avg_sentence_length_feature_set, \
        training_punctuation, punctuation_feature_set


if __name__ == "__main__":
    # THIS FILE WILL IMPORT THE TEXTS AND EXTRACT THE FEATURES, THEN SAVE THEM AS PICKLES
    # First argument specifies whether to extract english, spanish or Both
    # If empty, extract both

    t1 = time.time()

    # # UNCOMMENT WHEN READY TO USE ARGUMENTS FOR ENGLISH AND SPANISH
    # 0 == both
    # 1 == english
    # -1 == spanish 
    language = config.BOTH

    if len(argv) > 1:
        if argv[1].lower() == 'spanish':
            language = config.SPANISH
        elif argv[1].lower() == 'english':
            language = config.ENGLISH
        elif argv[1].lower() == 'both':
            pass
        else:
            print("Please choose a language as argument, either 'english' 'spanish' or 'both' ")

    # FILE EXTRACTION #


    # Import processed gutenberg texts from folder and place as a list [A, B, ... , C]
    if language >= 0:
        print("English File and Feature Extraction")

        en_processed_texts, en_authors = file_extraction(config.ENGLISH)

        # FEATURE EXTRACTION AND ASSEMBLY #
        en_char_ngram_feature_set, en_training_char_ngrams, \
            en_word_ngram_feature_set, en_training_word_ngrams, \
            en_avg_sentence_length_feature_set, \
            en_training_punctuation, en_punctuation_feature_set = feature_extraction(en_processed_texts, config.ENGLISH)
    
    if language <= 0:
        print("Spanish File and Feature Extraction")
        
        sp_processed_texts, sp_authors = file_extraction(config.SPANISH)

        # FEATURE EXTRACTION AND ASSEMBLY #
        sp_char_ngram_feature_set, sp_training_char_ngrams, \
            sp_word_ngram_feature_set, sp_training_word_ngrams, \
            sp_avg_sentence_length_feature_set, \
            sp_training_punctuation, sp_punctuation_feature_set = feature_extraction(sp_processed_texts, config.SPANISH)

    # FILE SAVING #
    t0 = time.time()

    # English
    if language >= 0:
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

        # save english average sentence length feature set
        with open(config.en_authors_path, 'wb') as fid:
            pickle.dump(en_authors, fid)

        # save english punctuation frequency feature set
        with open(config.en_punctuation_feature_set_path, 'wb') as fid:
            pickle.dump(en_punctuation_feature_set, fid)

        # save english punctuation training used for punctuation frequency
        with open(config.en_training_punctuation_path, 'wb') as fid:
            pickle.dump(en_training_punctuation, fid)

    # Spanish
    if language <= 0:
        # Save spanish training words used for character ngrams
        with open(config.sp_training_char_ngrams_path, 'wb') as fid:
            pickle.dump(sp_training_char_ngrams, fid)

        # Save spanish training ngrams used for word ngrams
        with open(config.sp_training_word_ngrams_path, 'wb') as fid:
            pickle.dump(sp_training_word_ngrams, fid)

        # Save spanish char ngram feature set
        with open(config.sp_char_ngram_feature_set_path, 'wb') as fid:
            pickle.dump(sp_char_ngram_feature_set, fid)

        # Save spanish word ngram feature set
        with open(config.sp_word_ngram_feature_set_path, 'wb') as fid:
            pickle.dump(sp_word_ngram_feature_set, fid)

        # save spanish average sentence length feature set
        with open(config.sp_avg_sentence_length_feature_set_path, 'wb') as fid:
            pickle.dump(sp_avg_sentence_length_feature_set, fid)

        # save spanish average sentence length feature set
            with open(config.sp_authors_path, 'wb') as fid:
                pickle.dump(sp_authors, fid)

        # save spanish punctuation frequency feature set
            with open(config.sp_punctuation_feature_set_path, 'wb') as fid:
                pickle.dump(sp_punctuation_feature_set, fid)

        # save spanish punctuation training used for punctuation frequency
            with open(config.sp_training_punctuation_path, 'wb') as fid:
                pickle.dump(sp_training_punctuation, fid)
    
    print("File Save Time:", time.time() - t0)

    # Output success
    if language >= 0:
        print(config.EN_NAME, 'features successfully extracted and saved in', time.time() - t1, 'seconds.')
    if language <= 0:
        print(config.SP_NAME, 'features successfully extracted and saved in', time.time() - t1, 'seconds.')
