# TWEAK CLASSIFIER OPTIONS
# If tweaking, you MUST run extract_classifier.py again
char_ngram_length = 4  # length of character ngrams
word_ngram_length = 2  # length of word ngrams
ngram_common_words = 100  # how many of the most common ngrams to choose as feature sets  
punctuation_length = 16  # how many of most common punctuations to choose as feature sets
normalization_type = 'l2' # approacah for normalisation
# L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
# L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration

# Paths for preprocessed texts
en_processed_text_path = './processed_texts/english'
sp_processed_text_path = './processed_texts/spanish'

# Paths for best classifiers for further testing
en_best_classifier_text_path = './classifier/best_classifiers.txt'
sp_best_classifier_text_path = './classifier/best_classifiers.txt'

# Paths for english classifiers and english serialised objects
en_classifier_path = './serialised_objects/en_classifier.pkl'
en_authors_path = './serialised_objects/en_authors.pkl'
en_training_char_ngrams_path = './serialised_objects/en_training_char_ngrams.pkl'
en_training_word_ngrams_path = './serialised_objects/en_training_word_ngrams.pkl'
en_char_ngram_feature_set_path = './serialised_objects/en_char_ngram_feature_set.pkl'
en_word_ngram_feature_set_path = './serialised_objects/en_word_ngram_feature_set.pkl'
en_avg_sentence_length_feature_set_path = './serialised_objects/en_avg_sentence_length_feature_set.pkl'
en_punctuation_feature_set_path = './serialised_objects/en_punctuation_feature_set.pkl'
en_training_punctuation_path = './serialised_objects/en_training_punctuation.pkl'
en_k_folds = 10

# Paths for spanish classifiers and and spanish serialised objects
sp_classifier_path = './serialised_objects/sp_classifier.pkl'
sp_authors_path = './serialised_objects/sp_authors.pkl'
sp_training_char_ngrams_path = './serialised_objects/sp_training_char_ngrams.pkl'
sp_training_word_ngrams_path = './serialised_objects/sp_training_word_ngrams.pkl'
sp_char_ngram_feature_set_path = './serialised_objects/sp_char_ngram_feature_set.pkl'
sp_word_ngram_feature_set_path = './serialised_objects/sp_word_ngram_feature_set.pkl'
sp_avg_sentence_length_feature_set_path = './serialised_objects/sp_avg_sentence_length_feature_set.pkl'
sp_punctuation_feature_set_path = './serialised_objects/sp_punctuation_feature_set.pkl'
sp_training_punctuation_path = './serialised_objects/sp_training_punctuation.pkl'
sp_k_folds = 8

# Other paths
experiment_results_path = './experiment_results/'

# Constants
SPANISH = -1
ENGLISH = 1
BOTH = 0
SP_NAME = 'Spanish'
EN_NAME = 'English'