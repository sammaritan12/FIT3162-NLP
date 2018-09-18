import unittest
from preprocessor import list_filenames, extract_text, prepend_parent_dir_to_file

class TestExtractTest(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, extract_text, 55)
        self.assertRaises(TypeError, extract_text, lambda x: x + 1)
        self.assertRaises(TypeError, extract_text, [23])
        self.assertRaises(TypeError, extract_text, (12, 23))
        self.assertRaises(TypeError, extract_text, 3.14159)

    def test_nonexisting_file(self):
        test_output = list_filenames("this file doesn't exist")
        self.assertEqual([], test_output)

    def test_valid_data(self):
        filename = 'test\\t2.txt'
        fd = open(filename)
        original_text = fd.read()
        valid = 'xyz\n'
        fd.close()

        extract_text(filename)

        fd = open(filename)
        new_text = fd.read()
        fd.close()

        self.assertEqual(new_text, valid)

        fd = open(filename, 'w')
        fd.write(original_text)
        fd.close()

class TestPrependParentDirToFile(unittest.TestCase):
    def test_types(self):
        self.assertRaises(TypeError, prepend_parent_dir_to_file, 55)
        self.assertRaises(TypeError, prepend_parent_dir_to_file, lambda x: x + 1)
        self.assertRaises(TypeError, prepend_parent_dir_to_file, [23])
        self.assertRaises(TypeError, prepend_parent_dir_to_file, (12, 23))
        self.assertRaises(TypeError, prepend_parent_dir_to_file, 3.14159)

    def test_nonexisting_file(self):
        test_output = list_filenames("this file doesn't exist")
        self.assertEqual([], test_output)

    def test_valid_data(self):
        prepend_parent_dir_to_file('test\\t1.txt')
        fd = open('test\\t1.txt', 'r+')
        top = fd.readline()
        fd.close()
        self.assertEqual(top, 'test\n')

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