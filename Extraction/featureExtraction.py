from nltk import sent_tokenize, word_tokenize, FreqDist
from os import listdir
from os.path import isfile, join

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extraction will be done using NLTK

### FREQDIST AUXILIARY FUNCTIONS ###
# [x] Merge multiple frequency distributions, and grab n most common items

def most_common(freqDists, n):
    """
    Create FreqDist from list of FreqDist which contain n most common words
    - freqDists is represented as [FreqDist, FreqDist, ... , FreqDist]
    - n is most common character
    """
    mergedFreqs = FreqDist()
    if len(freqDists) > 1:
        for i in range(len(freqDists)):
            mergedFreqs += freqDists[i]
            
    return mergedFreqs.most_common(n)


### NGRAM FEATURE EXTRACTION ###
# [x] Create ngram from list of words
# [] Find n most common ngrams from entire corpus
# [] Select n most common ngrams that occur in each text

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

### FREQUENCY DISTRIBUTION FEATURES ###
# [x] Average sentence length
# [] Punctuation frequency distribution
# [] POS tagging
# [] POS frequency distribution

# TODO punctuation feature extraction

# TODO POS tagging

# TODO POS frequency distribution

### PUTTING IT ALL TOGETHER ###
# Join features such that there consists 2 lists, x, y
# x: [Text Features A, Text Features B, ... , Text Features C]
# Text Features: [Average Sentence Length, 
#                 Punctuation 1 Freq, ... , Punctuation 14 Freq,
#                 POS 1 Freq, ... , POS 8 Freq,
#                 Most Common N-Gram Freq, ... , 1000th Most Common N-Gram Freq]
# y: [Author A, Author B, ... , Author C]

def avg_sentence_length(text):
    """
    Returns the median sentence length of a text
    - text is represented as a string 'text'
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    return len(words) / len(sentences)