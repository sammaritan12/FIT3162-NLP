from nltk import sent_tokenize, word_tokenize, FreqDist
from os import listdir
from os.path import isfile, join
from math import ceil

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extraction will be done using NLTK

### FREQDIST AUXILIARY FUNCTIONS ###
# [x] Merge multiple frequency distributions
# [x] Select n most common items from frequency distributions
# [x] Find n most common items for each frequency distribution from aggregated set

def merge_freqdists(freq_dists):
    """
    Merge multiple frequency lists together
    - freq_dists is represented as [FreqDist, FreqDist, ... , FreqDist]

    Returns
    - merged_freqs , a FreqDist of the joined freq_dists
    """
    merged_freqs = FreqDist()
    if len(freq_dists) > 1:
        for i in range(len(freq_dists)):
            merged_freqs += freq_dists[i]
            
    return merged_freqs

def freqdist_selection(text_distributions, n):
    """
    Selects the most common n words from text_distributions
    - text_distributions, list of FreqDists [A, B, ... , C]

    Returns
    - item , list containing actual values [A, B, ... , C]
    - item_counts , list containing counts of items [A count, B count, ... , C count]
    """
    # merge distributions
    merged_dist = merge_freqdists(text_distributions)

    # intialise item count list and items
    items = []
    item_counts = [[] for _ in range(len(text_distributions))]

    # get top n words, or size if n > length of merged_dist
    top_n_items = merged_dist.most_common(min(ceil(n), len(merged_dist)))

    # Append words and word counts to their lists
    for i in range(n):
        for t in range(len(text_distributions)):
            if text_distributions[t][top_n_items[i][0]]:
                item_counts[t].append(text_distributions[t][ top_n_items[i][0] ])
            else:
                item_counts[t].append(0)
        items.append(top_n_items[i][0])

    return items, item_counts

def freqdist_test_selection(distribution, items):
    """
    Selects item counts from distribution and places into a simple list
    """
    return [distribution[i] for i in items]

### NGRAM FEATURE EXTRACTION ###
# [x] Create character ngrams from list of words
# [x] Create word ngrams from list of words

def char_ngram(length, words):
    """
    Given a list of strings, convert to character ngrams, where length is length of ngram
    - words is represented as [a, b, ..., c.]
    - where a, b, c are whole words
    - length is the length of the character n-gram required

    Returns
    - ngrams , list of character ngrams within the text
    """
    ngrams = []
    for i in words:
        # if words is less than or equal to length, it is already an n-gram
        if len(i) <= length:
            ngrams.append(i)
        else:
            # else, split into n-grams for length characters
            for j in range(length - 1, len(i)):
                ngrams.append(i[j - length + 1: j + 1])
    return ngrams

def word_ngram(length, words):
    """
    Given a list of strings, convert to word ngrams, where length is length of ngram
    - words is represented as [a, b, ..., c.]
    - where a, b, c are whole words
    - length is the length of the word n-gram required

    Returns
    - ngrams , list of word ngrams within the text
    """
    ngrams = []
    # if words is less than or equal to length, it is already an n-gram
    if len(words) <= length:
        ngrams.append(" ".join(words))
    else:
        # else, split into n-grams for length words
        for i in range(length - 1, len(words)):
            ngrams.append(" ".join(words[i - length + 1: i + 1]))
    return ngrams

def ngram_selection(text_distributions, n):
    # merge distributions
    merged_dist = merge_freqs(text_distributions)
    # initialise word count list and words
    words = []
    word_counts = [[] for _ in range(len(text_distributions))]

    # get top n words, or size if n > length of merged_dist
    top_n_words = merged_dist.most_common(min(ceil(n), len(merged_dist)))
    # Append words and word counts to their lists
    for i in range(n):
        for t in range(len(text_distributions)):
            if text_distributions[t][top_n_words[i][0]]:
                word_counts[t].append(text_distributions[t][ top_n_words[i][0] ])
            else:
                word_counts[t].append(0)
        words.append(top_n_words[i][0])
    return words, word_counts

def ngram_test_selection(distribution, words):
    return [distribution[w] for w in words]

### FREQUENCY DISTRIBUTION FEATURES ###
# [x] Average sentence length
# [] Punctuation frequency distribution
# [] POS tagging and frequency distribution

# TODO punctuation feature extraction

# TODO POS tagging and freq distribution

def avg_sentence_length(text, words):
    """
    Returns the median sentence length of a text
    - text is represented as a string 'text'
    """
    sentences = sent_tokenize(text)
    # words = word_tokenize(text)

    return len(words) / len(sentences)

def punctuation_frequency(text_distributions):
    punctuation_set = [".", "?", "!", ",", ";", ":", "−", "-", "[", "]", "{", "}", "(", ")", "'", "\""]
    # initialise word count list and words
    words = []
    word_counts = [[] for _ in range(len(text_distributions))]

    # Append words and word counts to their lists
    for i in range(len(punctuation_set)):
        for t in range(len(text_distributions)):
            if text_distributions[t][punctuation_set[i][0]]:
                word_counts[t].append(text_distributions[t][punctuation_set[i][0] ])
            else:
                word_counts[t].append(0)
        words.append(punctuation_set[i])
    return words, word_counts