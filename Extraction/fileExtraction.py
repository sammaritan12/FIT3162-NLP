import glob
### FILE EXTRACTION ###
# [X] Get filename, convert to text
# [X] Get filenames of all files into list
# [x] Find way to get author from text (Amelia)
# [] Convert all text files in folder to 2 lists (texts and authors)

def filename_to_text(filename):
    """Given filename, convert to a string and return string"""
    file_object = open(filename, 'r')
    text = file_object.read()
    file_object.close()
    return text

def list_filenames(dirpath):
    return glob.glob(dirpath + '/**/*.txt', recursive=True)
# TODO Extract author from text

# TODO Create text and author list