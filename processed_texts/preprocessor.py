from timeit import default_timer as timer
from glob import glob
from sys import argv
import os

def list_filenames(dirpath):
    '''
    Grabs all text files in diretory and returns as list
    Params:
    - dirpath, string for directory path
    '''
    if type(dirpath) is not str:
        raise TypeError("dirpath must be a string path to directory")

    return glob(dirpath + '/**/*.txt', recursive=True)

def extract_text(filename):
    '''
    Grabs text from Gutenberg Project and takes out unnecesary words
    Params
    - filename, string for filename
    '''
    if type(filename) is not str:
        raise TypeError("dirpath must be a string path to directory")
    
    # Go through the file and find where the file starts and ends, and only extract those
    # Tries not to extract meta information
    start = timer()
    file = open(filename, encoding='utf-8', errors='ignore')
    file_lines = []
    i = 0

    for line in file:
        if line.strip() != "":
            file_lines.append(line.strip())
        i += 1
    start_found = False
    start_line = 0
    end_line = -1

    for i in range(len(file_lines)):
        if file_lines[i].startswith("***"):
            if not start_found:
                start_line = i
                start_found = True
            else:
                end_line = i-1
                break

    file_lines = file_lines[start_line+1:end_line]

    file.close()
    file = open(filename, "w")

    for line in file_lines:
        file.write(line + "\n")

    file.close()
    end = timer()
    print("File processed in " + str((end-start)/1000) + " seconds")

def prepend_parent_dir_to_file(filename):
    '''
    Prepend file with parent directory name
    Params:
    - filename, string of path to file
    '''
    if type(filename) is not str:
        raise TypeError("dirpath must be a string path to directory")

    # Adds the folder name to the very top of path
    parent_dir_name = os.path.basename(os.path.dirname(filename))
    with open(filename, "r+") as file:
        content = file.read()
        file.seek(0,0)
        file.write(parent_dir_name.rstrip('\r\n') + '\n' + content)

if __name__ == '__main__':
    '''
    Function used to preprocess in file all items in the specified directory
    Run as:
    python preprocessor.py <directory_name>
    All files within directory_name must be Gutenberg texts
    '''
    # Make sure there's an argument
    if len(argv) < 2:
        print("Enter command as: preprocessor.py <directory_name>")
        quit()

    # Gets all files in directory, processes and places author in it
    filenames = list_filenames(argv[1])
    for i in filenames:
        extract_text(i)
        prepend_parent_dir_to_file(i)
