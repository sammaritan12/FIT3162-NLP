from glob import glob

def filename_to_text(filename):
    """Given filename, convert to a string and return string"""
    if type(filename) is not str:
        raise TypeError('Filename must be a string')

    file_object = open(filename, 'r')
    text = file_object.read()
    file_object.close()
    return text


def list_filenames(dirpath):
    if type(dirpath) is not str:
        raise TypeError('dirpath must be a string')

    return glob(dirpath + '/**/*.txt', recursive=True)

print(list_filenames('this dont exist'))