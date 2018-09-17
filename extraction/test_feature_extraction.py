import unittest

from nltk import word_tokenize, FreqDist

from feature_extraction import (avg_sentence_length, char_ngram,
                                freqdist_selection, freqdist_test_selection,
                                merge_freqdists, punctuation_frequency,
                                word_ngram)


# class TestAvgSentenceLength(unittest.TestCase):
#     def test(self):
#         pass

# class TestCharNgram(unittest.TestCase):
#     def test(self):
#         pass

# class TestWordNgram(unittest.TestCase):
#     def test(self):
#         pass

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

# class TestFreqDistSelection(unittest.TestCase):
#     def test(self):
#         pass

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

# class TestFreqDistTestSelection(unittest.TestCase):
#     def test(self):
#         pass

if __name__ == '__main__':
    unittest.main()
