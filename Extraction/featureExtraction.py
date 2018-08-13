from nltk import sent_tokenize, word_tokenize, FreqDist
from os import listdir
from os.path import isfile, join
from math import ceil

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extraction will be done using NLTK

### FREQDIST AUXILIARY FUNCTIONS ###
# [x] Merge multiple frequency distributions

def merge_freqs(freqDists):
    """
    Merge multiple frequency lists together
    - freqDists is represented as [FreqDist, FreqDist, ... , FreqDist]
    """
    mergedFreqs = FreqDist()
    if len(freqDists) > 1:
        for i in range(len(freqDists)):
            mergedFreqs += freqDists[i]
            
    return mergedFreqs

### NGRAM FEATURE EXTRACTION ###
# [x] Create ngram from list of words
# [x] Find n most common ngrams from entire corpus
# [x] Select n most common ngrams that occur in each text

def char_ngram(length, words):
    """
    Given a list of strings, convert to character ngrams, where length is length of ngram
    - words is represented as [a, b, ..., c.]
    - where a, b, c are whole words
    - length is the length of the character n-gram required
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

def ngram_selection(text_distributions, n):
    # merge distributions
    merged_dist = merge_freqs(text_distributions)

    # intialise word count list and words
    words = []
    word_counts = [[] for _ in range(n)]

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
# [] POS tagging
# [] POS frequency distribution

# TODO punctuation feature extraction

# TODO POS tagging and freq distribution

def avg_sentence_length(text):
    """
    Returns the median sentence length of a text
    - text is represented as a string 'text'
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    return len(words) / len(sentences)

