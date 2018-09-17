from file_extraction import filename_to_text, list_filenames
import unittest

class TestFilenameToText(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, filename_to_text, 55)
        self.assertRaises(TypeError, filename_to_text, lambda x: x + 1)
        self.assertRaises(TypeError, filename_to_text, [23])
        self.assertRaises(TypeError, filename_to_text, (12, 23))
        self.assertRaises(TypeError, filename_to_text, 3.14159)

    def test_non_existing_file(self):
        self.assertRaises(FileNotFoundError, filename_to_text, 'file doesnt exists')

    def test_file_value(self):
        hi = open('test_filename_to_text.txt', 'w+')
        hi.write("This is a test")
        hi.close()

        self.assertEqual(filename_to_text('test_filename_to_text.txt'), "This is a test")

class TestListFilenames(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, list_filenames, 55)
        self.assertRaises(TypeError, list_filenames, lambda x: x + 1)
        self.assertRaises(TypeError, list_filenames, [23])
        self.assertRaises(TypeError, list_filenames, (12, 23))
        self.assertRaises(TypeError, list_filenames, 3.14159)

    def test_nonexisting_file(self):
        test_output = list_filenames("this file doesn't exist")
        self.assertEqual([], test_output)
    
    def test_existing_folder(self):
        test_output = list_filenames('test')
        self.assertEqual(['test\\t1.txt', 'test\\t2.txt'], test_output)

if __name__ == '__main__':
    unittest.main()