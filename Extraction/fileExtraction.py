### FILE EXTRACTION ###
# [X] Get filename, convert to text
# [X] Get filenames of all files into list
# [] Find way to get author from text (Amelia)
# [] Convert all text files in folder to 2 lists (texts and authors)

def filename_to_text(filename):
    """Given filename, convert to a string and return string"""
    file_object = open(filename, 'r')
    text = file_object.read()
    file_object.close()
    return text

def list_filenames(dirpath):
    """Returns a list of all files in a given directory path"""
    filenames = []
    for f in listdir(dirpath):
        if isfile(join(dirpath, f)):
            filenames.append(f)
    return filenames

# TODO Extract author from text

# TODO Create text and author list