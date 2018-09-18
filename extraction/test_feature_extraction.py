import unittest

from nltk import word_tokenize, FreqDist

from feature_extraction import (avg_sentence_length, char_ngram,
                                freqdist_selection, freqdist_test_selection,
                                merge_freqdists, punctuation_frequency,
                                word_ngram)


class TestAvgSentenceLength(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, avg_sentence_length, -23, [''])
        self.assertRaises(TypeError, avg_sentence_length, {}, [''])
        self.assertRaises(TypeError, avg_sentence_length, -23, [''])
        self.assertRaises(TypeError, avg_sentence_length, lambda x: x, [''])
        self.assertRaises(TypeError, avg_sentence_length, [34], [''])

        self.assertRaises(TypeError, avg_sentence_length, 'a', 12)
        self.assertRaises(TypeError, avg_sentence_length, 'a', {23})
        self.assertRaises(TypeError, avg_sentence_length, 'a', (1,2))
        self.assertRaises(TypeError, avg_sentence_length, 'a', lambda x: x)
        self.assertRaises(TypeError, avg_sentence_length, 'a', 3.14)

    def test_malformed_data(self):
        # non strings in words
        self.assertRaises(ValueError, avg_sentence_length, 'asdfg', ['hi', 34])

        # empty text
        self.assertRaises(ValueError, avg_sentence_length, '', ['hi'])

    def test_valid_data(self):
        self.assertEqual(avg_sentence_length('hello', ['hello']), 1)


class TestCharNgram(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, char_ngram, [55], ['hi'])
        self.assertRaises(TypeError, char_ngram, lambda x: x + 1, ['hi'])
        self.assertRaises(TypeError, char_ngram, "i shouldn't be a string", ['hi'])
        self.assertRaises(TypeError, char_ngram, (12, 23), ['hi'])
        self.assertRaises(TypeError, char_ngram, 3.14159, ['hi'])

        self.assertRaises(TypeError, char_ngram, 55, {})
        self.assertRaises(TypeError, char_ngram, 55, lambda x: x + 1)
        self.assertRaises(TypeError, char_ngram, 55, "i shouldn't be a string")
        self.assertRaises(TypeError, char_ngram, 55, (12, 23))
        self.assertRaises(TypeError, char_ngram, 55, 3.14159)

    def test_malformed_data(self):
        self.test_words = ['the', [23],'qui', 43,'ick']
        self.assertRaises(ValueError, char_ngram, 4, self.test_words)

    def test_correct_n(self):
        self.assertRaises(ValueError, char_ngram, 0, ['the', 'qui', 'uic', 'ick'])
        self.assertRaises(ValueError, char_ngram, -2, ['the', 'qui', 'uic', 'ick'])

    def test_valid_data(self):
        self.test_output = ['the', 'hi']
        self.valid = ['t', 'h', 'e', 'h', 'i']
        # for 1 grams
        self.assertEqual(char_ngram(1, self.test_output), self.valid)

        # for when the character length is longer than all strings
        self.assertEqual(char_ngram(999, self.test_output), self.test_output)

class TestWordNgram(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, word_ngram, [55], ['hi'])
        self.assertRaises(TypeError, word_ngram, lambda x: x + 1, ['hi'])
        self.assertRaises(TypeError, word_ngram, "i shouldn't be a string", ['hi'])
        self.assertRaises(TypeError, word_ngram, (12, 23), ['hi'])
        self.assertRaises(TypeError, word_ngram, 3.14159, ['hi'])

        self.assertRaises(TypeError, word_ngram, 55, {})
        self.assertRaises(TypeError, word_ngram, 55, lambda x: x + 1)
        self.assertRaises(TypeError, word_ngram, 55, "i shouldn't be a string")
        self.assertRaises(TypeError, word_ngram, 55, (12, 23))
        self.assertRaises(TypeError, word_ngram, 55, 3.14159)

    def test_malformed_data(self):
        self.test_words = ['the', [23],'qui', 43,'ick']
        self.assertRaises(ValueError, word_ngram, 4, self.test_words)

    def test_correct_n(self):
        self.assertRaises(ValueError, word_ngram, 0, ['the', 'qui', 'uic', 'ick'])
        self.assertRaises(ValueError, word_ngram, -2, ['the', 'qui', 'uic', 'ick'])

    def test_valid_data(self):
        self.test_output = ['the', 'hi']
        self.valid = ['the hi']
        # for 1 grams
        self.assertEqual(word_ngram(2, self.test_output), self.valid)

        # for when the word length is longer than all strings
        self.assertEqual(word_ngram(999, self.test_output), self.valid)

class TestPunctuationFreqency(unittest.TestCase):
    def test_correct_output(self):
        test_sentence = 'the, quick brown fox. jumps over the lazy dog!'
        test_output = punctuation_frequency(word_tokenize(test_sentence))
        self.assertEqual(test_output, [',', '.', '!'])

    def test_types(self):
        self.assertRaises(TypeError, punctuation_frequency, 55)
        self.assertRaises(TypeError, punctuation_frequency, lambda x: x + 1)
        self.assertRaises(TypeError, punctuation_frequency, "i shouldn't be a string")
        self.assertRaises(TypeError, punctuation_frequency, (12, 23))
        self.assertRaises(TypeError, punctuation_frequency, 3.14159)

    def test_malformed_data(self):
        test_input = ['.', 23]
        self.assertRaises(ValueError, punctuation_frequency, test_input)

class TestFreqDistSelection(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, freqdist_selection, 55, 5)
        self.assertRaises(TypeError, freqdist_selection, lambda x: x + 1, 5)
        self.assertRaises(TypeError, freqdist_selection, "i shouldn't be a string", 5)
        self.assertRaises(TypeError, freqdist_selection, (12, 23), 5)
        self.assertRaises(TypeError, freqdist_selection, 3.14159, 5)
        
        self.assertRaises(TypeError, freqdist_selection, [FreqDist()], '5')
        self.assertRaises(TypeError, freqdist_selection, [FreqDist()], (34, 5))
        self.assertRaises(TypeError, freqdist_selection, [FreqDist()], [3])
        self.assertRaises(TypeError, freqdist_selection, [FreqDist()], lambda x: x)
        self.assertRaises(TypeError, freqdist_selection, [FreqDist()], {})

    def test_malformed_data(self):
        self.assertRaises(ValueError, freqdist_selection, [FreqDist(), 45], 45)

    def test_weird_n(self):
        test_ouput = [FreqDist(word_tokenize('the quick brown brown fox'))]
        self.assertRaises(ValueError, freqdist_selection, test_ouput, 0)
        self.assertRaises(ValueError, freqdist_selection, test_ouput, -45)
        self.assertRaises(TypeError, freqdist_selection, test_ouput, 4.3)
    
class TestMergeFreqDists(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, punctuation_frequency, 55)
        self.assertRaises(TypeError, punctuation_frequency, lambda x: x + 1)
        self.assertRaises(TypeError, punctuation_frequency, "i shouldn't be a string")
        self.assertRaises(TypeError, punctuation_frequency, (12, 23))
        self.assertRaises(TypeError, punctuation_frequency, 3.14159)
        self.assertRaises(TypeError, punctuation_frequency, FreqDist)

    def test_malformed_data(self):
        test_output = [FreqDist(), 23, FreqDist()]
        self.assertRaises(ValueError, merge_freqdists, test_output)

    def test_valid_data(self):
        t1 = FreqDist(word_tokenize('the quick brown fox jumps'))
        t2 = FreqDist(word_tokenize('the quick brown fox jumps'))
        self.assertEqual(merge_freqdists([t1, t2]), t1 + t2)

class TestFreqDistTestSelection(unittest.TestCase):
    def test_types(self):
        self.test_distribution = FreqDist(word_tokenize('the the the quick brown'))
        self.test_items = ['the', 'quick', 'brown']
        self.correct_output = [3, 1, 1]

        # testing distribution data types
        self.assertRaises(TypeError, freqdist_test_selection, 55, self.test_items)
        self.assertRaises(TypeError, freqdist_test_selection, lambda x: x + 1, self.test_items)
        self.assertRaises(TypeError, freqdist_test_selection, "i shouldn't be a string", self.test_items)
        self.assertRaises(TypeError, freqdist_test_selection, (12, 23), self.test_items)
        self.assertRaises(TypeError, freqdist_test_selection, 3.14159, self.test_items)
        self.assertRaises(TypeError, freqdist_test_selection, [34], self.test_items)

        # testing items data types
        self.assertRaises(TypeError, freqdist_test_selection, FreqDist(), 55)
        self.assertRaises(TypeError, freqdist_test_selection, FreqDist(), lambda x: x + 1)
        self.assertRaises(TypeError, freqdist_test_selection, FreqDist(), "i shouldn't be a string")
        self.assertRaises(TypeError, freqdist_test_selection, FreqDist(), (12, 23))
        self.assertRaises(TypeError, freqdist_test_selection, FreqDist(), 3.14159)

    def test_malformed_data(self):
        # items must be a list of strings
        self.test_distribution = FreqDist(word_tokenize('the the the quick brown'))
        self.test_items = ['the', 'quick', 'brown']
        self.correct_output = [3, 1, 1]

        self.assertRaises(ValueError, freqdist_test_selection, self.test_distribution, ['the', 34, [34]])
    
    def test_valid_data(self):
        # testing whether output is correct
        self.test_distribution = FreqDist(word_tokenize('the the the quick brown'))
        self.test_items = ['the', 'quick', 'brown']
        self.correct_output = [3, 1, 1]
        
        self.assertEqual(freqdist_test_selection(self.test_distribution,self.test_items), self.correct_output)

if __name__ == '__main__':
    unittest.main()
