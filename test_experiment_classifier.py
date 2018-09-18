import unittest
from experiment_classifier import cv_classifier_score, test_classifier_kernels
from sklearn.svm import LinearSVC


class TestCVClassifierScore(unittest.TestCase):
    def test_types(self):
        # test k fold param
        self.assertRaises(TypeError, cv_classifier_score, LinearSVC(), [[1]], [1], 2.5)
        self.assertRaises(TypeError, cv_classifier_score, LinearSVC(), [[1]], [1], '2.5')
        self.assertRaises(TypeError, cv_classifier_score, LinearSVC(), [[1]], [1], [2.5,3])
        self.assertRaises(TypeError, cv_classifier_score, LinearSVC(), [[1]], [1], (2.5, 3))

    def test_positive_k_fold(self):
        self.assertRaises(ValueError, cv_classifier_score, LinearSVC(), [[1]], [1], 0)
        self.assertRaises(ValueError, cv_classifier_score, LinearSVC(), [[1]], [1], -2)

class TestTestClassifierKernels(unittest.TestCase):
    def test_types(self):
        author = ['Mark', 'Twain']
        features = [[1,1,0,0,1,1], [0,0,0,1,1,1,]]
        fe_text = 'text'
        k = 1
        lang = 1

        # language
        self.assertRaises(TypeError, test_classifier_kernels, author, features, fe_text, 'lang', k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, fe_text, [lang], k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, fe_text, lambda x: x, k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, fe_text, (lang, lang), k)

        # feature_text
        self.assertRaises(TypeError, test_classifier_kernels, author, features, 12, lang, k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, [fe_text], lang, k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, (fe_text, fe_text), lang, k)
        self.assertRaises(TypeError, test_classifier_kernels, author, features, lambda x: x, lang, k)

    def test_lang(self):
        author = ['Mark', 'Twain']
        features = [[1,1,0,0,1,1], [0,0,0,1,1,1,]]
        fe_text = 'text'
        k = 1

        self.assertRaises(ValueError, test_classifier_kernels, author, features, fe_text, 0, k)
        self.assertRaises(ValueError, test_classifier_kernels, author, features, fe_text, 2, k)
        self.assertRaises(ValueError, test_classifier_kernels, author, features, fe_text, -2, k)
        self.assertRaises(ValueError, test_classifier_kernels, author, features, fe_text, 500, k)

if __name__ == '__main__':
    unittest.main()