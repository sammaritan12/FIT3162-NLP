from Extraction.featureExtraction import *
from Classifier.englishClassifier import *
from Extraction.fileExtraction import *
from sklearn.preprocessing import normalize
import pickle
import config
from sys import argv

if __name__ == "__main__":
    # THIS FILE WILL CREATE AND FIT THE CLASSIFIER, THEN SAVE CLASSIFIER
    # First argument specifies whether to build English, Spanish or Both
    # If empty, build both

    # TWEAK CLASSIFIER OPTIONS
    ngram_length = 4  # length of character ngrams
    ngram_common_words = 5  # how many of the most common ngrams to choose as feature sets  
    normalization_type = 'l1'
        # L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
        # L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration

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

    # TODO Import raw gutenberg texts from folder and place as a list [A, B, ... , C]

    text_a = '''There were a king with a large jaw and a queen with a plain face, on the
    throne of England; there were a king with a large jaw and a queen with
    a fair face, on the throne of France. In both countries it was clearer
    than crystal to the lords of the State preserves of loaves and fishes,
    that things in general were settled for ever.'''

    text_b = '''All these things, and a thousand like them, came to pass in and close
    upon the dear old year one thousand seven hundred and seventy-five.
    Environed by them, while the Woodman and the Farmer worked unheeded,
    those two of the large jaws, and those other two of the plain and the
    fair faces, trod with stir enough, and carried their divine rights
    with a high hand. Thus did the year one thousand seven hundred
    and seventy-five conduct their Greatnesses, and myriads of small
    creatures--the creatures of this chronicle among the rest--along the
    roads that lay before them.'''

    text_c = '''With drooping heads and tremulous tails, they mashed their way through
    the thick mud, floundering and stumbling between whiles, as if they were
    falling to pieces at the larger joints. As often as the driver rested
    them and brought them to a stand, with a wary “Wo-ho! so-ho-then!” the
    near leader violently shook his head and everything upon it--like an
    unusually emphatic horse, denying that the coach could be got up the
    hill. Whenever the leader made this rattle, the passenger started, as a
    nervous passenger might, and was disturbed in mind.'''

    text_d = '''There was a steaming mist in all the hollows, and it had roamed in its
    forlornness up the hill, like an evil spirit, seeking rest and finding
    none. A clammy and intensely cold mist, it made its slow way through the
    air in ripples that visibly followed and overspread one another, as the
    waves of an unwholesome sea might do. It was dense enough to shut out
    everything from the light of the coach-lamps but these its own workings,
    and a few yards of road; and the reek of the labouring horses steamed
    into it, as if they had made it all.'''

    text_e = '''The Dover mail was in its usual genial position that the guard suspected
    the passengers, the passengers suspected one another and the guard, they
    all suspected everybody else, and the coachman was sure of nothing but
    the horses; as to which cattle he could with a clear conscience have
    taken his oath on the two Testaments that they were not fit for the
    journey.'''

    texts = [text_a, text_b, text_c, text_d, text_e]

    # TODO Preprocess raw gutenberg texts
    # Extract only needed text parts and find author name within text

    # TODO Extract feature set and authors from gutenberg texts

    # Extract authors
    authors = ['Author A', 'Author B', 'Author C', 'Author D', 'Author E']

    # Extract character ngrams from text
    ngram_text_dists = [FreqDist(char_ngram(config.ngram_length, word_tokenize(i))) for i in texts]
    # collate ngram most common ngrams and their ocurrences
    training_words, ngram_feature_set = ngram_selection(ngram_text_dists, config.ngram_common_words)

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
    training_feature_set_normalised = normalize(training_feature_set, norm=normalization_type)

    # Fit classifier
    eng_classifier = englishClassifier(training_feature_set_normalised, authors)

    # Save the classifier
    with open('eng_classifier.pkl', 'wb') as fid:
        pickle.dump(eng_classifier, fid)

    # Save training words used for ngrams
    with open('eng_training_words.pkl', 'wb') as fid:
        pickle.dump(training_words, fid)
    
    # Output success
    print('Classifier Successfully Created and Saved.')
