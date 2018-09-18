from timeit import default_timer as timer
from glob import glob
from sys import argv
import os

def list_filenames(dirpath):
    '''Grabs all text files in diretory and returns as list'''
    if type(dirpath) is not str:
        raise TypeError("dirpath must be a string path to directory")

    return glob(dirpath + '/**/*.txt', recursive=True)

def extract_text(filename):
    '''Grabs text from Gutenberg Project and takes out unnecesary words'''
    if type(filename) is not str:
        raise TypeError("dirpath must be a string path to directory")
    
    start = timer()
    file = open(filename, encoding='utf-8', errors='ignore')
    fileLines = []
    i = 0
    for line in file:
        if line.strip() != "":
            fileLines.append(line.strip())
        i += 1
    startFound = False
    startLine = 0
    endLine = -1
    for i in range(len(fileLines)):
        if fileLines[i].startswith("***"):
            if not startFound:
                startLine = i
                startFound = True
            else:
                endLine = i-1
                break
    fileLines = fileLines[startLine+1:endLine]
    file.close()
    file = open(filename, "w")
    for line in fileLines:
        file.write(line + "\n")
    file.close()
    end = timer()
    print("File processed in " + str((end-start)/1000) + " seconds")

def prepend_parent_dir_to_file(filename):
    '''Prepend file with parent directory name'''
    if type(filename) is not str:
        raise TypeError("dirpath must be a string path to directory")

    parent_dir_name = os.path.basename(os.path.dirname(filename))
    with open(filename, "r+") as file:
        content = file.read()
        file.seek(0,0)
        file.write(parent_dir_name.rstrip('\r\n') + '\n' + content)

if __name__ == '__main__':
    # Make sure there's an argument
    if len(argv) < 2:
        print("Enter command as: preprocessor.py <directory_name>")
        quit()

    # Gets all files in directory, processes and places author in it
    filenames = list_filenames(argv[1])
    for i in filenames:
        extract_text(i)
        prepend_parent_dir_to_file(i)
