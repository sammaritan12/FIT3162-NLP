# TWEAK CLASSIFIER OPTIONS
ngram_length = 4  # length of character ngrams
ngram_common_words = 1000  # how many of the most common ngrams to choose as feature sets  
normalization_type = 'l1' # approacah for normalisation
    # L1 Least Absolute Deviations, abs(sum of row) = 1, insensitive to outliers
    # L2 Least Squares, sum of squares, on each row = 1, takes outliers into consideration
processed_text_path = '.\ProcessedTexts'