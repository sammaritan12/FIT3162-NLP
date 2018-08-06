from nltk import sent_tokenize, word_tokenize, FreqDist

# Assumptions:
# - clean preprocessed data is a text file containing text only
# - extractionm will be done using NLTK

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


# TODO word n-gram feature extraction



# TODO punctuated feature extraction

# TODO average sentence length feature extraction
# this might be somewhat easy
# tokenize sentences, get average length (of whole piece or per amount of characters?)

def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    return len(words) / len(sentences)

# TODO frequency distribution of texts

def freq_dist(text):
    words = word_tokenize(text)
    return FreqDist(words)

fd = freq_dist('hello my name is Mark what is your name? I realise that the best way is the the the.')
for word, frequency in fd.most_common(50):
    print(u'{}; {}'.format(word, frequency))