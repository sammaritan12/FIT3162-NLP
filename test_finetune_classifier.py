import unittest

from finetune_classifier import finetune, test_linearsvc, test_randomforest

class TestFinetune(unittest.TestCase):
    def test_types(self):
        not_lang = 3.4
        self.assertRaises(TypeError, finetune, not_lang, [], [], [])

    def test_malformed_data(self):
        not_lang = 0
        self.assertRaises(ValueError, finetune, not_lang, [], [], [])

class TestTestLinearSVC(unittest.TestCase):
    def test_types(self):
        not_file = 'hi'
        self.assertRaises(TypeError, test_linearsvc, [], [], not_file, 0)
        
class TestTestRandomForest(unittest.TestCase):
    def test_types(self):
        not_file = 'hi'
        self.assertRaises(TypeError, test_linearsvc, [], [], not_file, 0)

if __name__ == '__main__':
    unittest.main()