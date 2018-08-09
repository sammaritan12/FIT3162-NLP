from nltk import sent_tokenize, word_tokenize, FreqDist

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extraction will be done using NLTK

# TODO import text files

def filename_to_text(filename):
    # open file and place entire text to string and return it
    file_object = open(filename, 'r')
    text = file_object.read()
    file_object.close()
    return text

# TODO character n-gram feature extraction
# space free character 4-gram
# a)string of characters of length 4 that includes no spaces
# b) string of characters 4 or less surrounded by spaces
# Take 1000 most common ngrams

def charNgram(length, words):
    # words is represented as [a, b, ..., c.]
    # where a, b, c are whole words
    # length is the length of the character n-gram required
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

# TODO punctuated feature extraction

# TODO average sentence length feature extraction
# this might be somewhat easy
# tokenize sentences, get average length (of whole piece or per amount of characters?)

def avg_sentence_length(text):
    # text is represented as a string 'text'
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    return len(words) / len(sentences)

# TODO frequency distribution of texts
def freq_dist(text):
    # text is represented as a string 'text'
    words = word_tokenize(text)
    return FreqDist(words)