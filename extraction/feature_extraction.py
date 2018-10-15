from math import ceil

from nltk import FreqDist, sent_tokenize, word_tokenize

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extraction will be done using NLTK

def merge_freqdists(freq_dists):
    """
    Merge multiple frequency lists together
    Params:
    - freq_dists is represented as [FreqDist, FreqDist, ... , FreqDist]

    Returns
    - merged_freqs , a FreqDist of the joined freq_dists
    """
    # check types
    if type(freq_dists) is not list:
        raise TypeError("freq_dists must be a list of FreqDists")

    # Basically add together freqdists
    merged_freqs = FreqDist()
    if len(freq_dists) > 1:
        for i in range(len(freq_dists)):

            if type(freq_dists[i]) is not FreqDist:
                raise ValueError("Each item in list must be FreqDist")

            merged_freqs += freq_dists[i]
            
    return merged_freqs


def freqdist_selection(text_distributions, n):
    """
    Selects the most common n words from text_distributions
    Params:
    - text_distributions, list of FreqDists [A, B, ... , C]
    - n, positive integer in 

    Returns
    - item , list containing actual values [A, B, ... , C]
    - item_counts , list containing counts of items [A count, B count, ... , C count]
    """
    # type and value checking
    if type(text_distributions) is not list:
        raise TypeError("text_distributions must be a list of FreqDists")

    if type(n) is not int:
        raise TypeError("N must be a integer")

    if n <= 0:
        raise ValueError("n must be a natural integer")

    if not all(isinstance(x, FreqDist) for x in text_distributions):
        raise ValueError("text_distribution must be a list of FreqDist data type")

    # merge distributions
    merged_dist = merge_freqdists(text_distributions)

    # initialise item count list and items
    items = []
    item_counts = [[] for _ in range(len(text_distributions))]

    # get top n words, or size if n > length of merged_dist
    top_n_items = merged_dist.most_common(min(ceil(n), len(merged_dist)))
    
    # Append words and word counts to their lists
    for i in range(min(ceil(n), len(merged_dist))):
        for t in range(len(text_distributions)):
            if text_distributions[t][top_n_items[i][0]]:
                item_counts[t].append(text_distributions[t][top_n_items[i][0]])
            else:
                item_counts[t].append(0)
        items.append(top_n_items[i][0])

    return items, item_counts


def freqdist_test_selection(distribution, items):
    """
    Selects item counts from distribution and places into a simple list
    Params:
    - distribution, FreqDist object
    - items, list of strings to check in the distribution param
    """
    # type and value checking
    if type(distribution) is not FreqDist:
        raise TypeError("distribution must be a FreqDist data type")

    if type(items) is not list:
        raise TypeError("items must be a list")

    if not all(isinstance(x, str) for x in items):
        raise ValueError("items must be a list of strings")        

    return [distribution[i] for i in items]


def char_ngram(length, words):
    """
    Given a list of strings, convert to character ngrams, where length is length of ngram
    Params:
    - words is represented as [a, b, ..., c.]
    - where a, b, c are whole words
    - length is the length of the character n-gram required

    Returns
    - ngrams , list of character ngrams within the text
    """
    # type and value checking
    if type(length) is not int:
        raise TypeError("Length must be an integer")

    if type(words) is not list:
        raise TypeError("Words must be a list of strings")

    if not all(isinstance(x, str) for x in words):
        raise ValueError("All items in words must be strings")

    if length <= 0:
        raise ValueError("Length must be a positive integer")

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
    Params:
    - words is represented as [a, b, ..., c.]
    - where a, b, c are whole words
    - length is the length of the word n-gram required

    Returns
    - ngrams , list of word ngrams within the text
    """
    # type and value checking
    if type(length) is not int:
        raise TypeError("Length must be an integer")

    if type(words) is not list:
        raise TypeError("Words must be a list of strings")

    if not all(isinstance(x, str) for x in words):
        raise ValueError("All items in words must be strings")

    if length <= 0:
        raise ValueError("Length must be a positive integer")


    ngrams = []
    # if words is less than or equal to length, it is already an n-gram
    if len(words) <= length:
        ngrams.append(" ".join(words))
    else:
        # else, split into n-grams for length words
        for i in range(length - 1, len(words)):
            ngrams.append(" ".join(words[i - length + 1: i + 1]))
    return ngrams


def avg_sentence_length(text, words):
    """
    Returns the median sentence length of a text
    Params:
    - text is represented as a string 'text'
    - words, tokenised list of strings within the text
    """
    # type and value checking
    if type(text) is not str:
        raise TypeError("Text must be a string")

    if type(words) is not list:
        raise TypeError("words must be a list")

    if len(text) <= 0:
        raise ValueError("text must be a nonempty string")

    if not all(isinstance(x, str) for x in words):
        raise ValueError("words must be a list of strings")

    sentences = sent_tokenize(text)
    # words = word_tokenize(text)

    return len(words) / len(sentences)


def punctuation_frequency(tokenised_words):
    """
    Returns the set of punctuation used in tokenised_words
    Params:
    - tokenised_words, list of strings of all words tokenised
    """
    # type and value checking
    if type(tokenised_words) is not list:
        raise TypeError('Input must be a list of strings')

    punctuation_set = [".", "?", "!", ",", ";", ":", "−", "-", "[", "]", "{", "}", "(", ")", "'", "\""]

    # initialise word count list and words
    result = []

    # Add punctuation to result
    for word in tokenised_words:
        if type(word) is not str:
            raise ValueError('Each list item must be a string')

        if word in punctuation_set:
            result.append(word)
    return result
