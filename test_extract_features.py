import unittest

from extract_features import feature_extraction, file_extraction


class TestFileExtraction(unittest.TestCase):
    def test_malformed_data(self):
        self.assertRaises(ValueError, file_extraction, 2)
        self.assertRaises(ValueError, file_extraction, -2)
        self.assertRaises(ValueError, file_extraction, 0)

class TestFeatureExtraction(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, feature_extraction, 23, 1)
        self.assertRaises(TypeError, feature_extraction, -23.3, 1)
        self.assertRaises(TypeError, feature_extraction, '23', 1)
        self.assertRaises(TypeError, feature_extraction, {}, 1)
        self.assertRaises(TypeError, feature_extraction, (1, 23), 1)

    def test_malformed_data(self):
        self.assertRaises(ValueError, feature_extraction, ['a'], 2)
        self.assertRaises(ValueError, feature_extraction, ['a'], -2)
        self.assertRaises(ValueError, feature_extraction, ['a'], 0)

        malformed = ['a', 2]
        self.assertRaises(ValueError, feature_extraction, malformed, 1)

if __name__ == '__main__':
    unittest.main()
