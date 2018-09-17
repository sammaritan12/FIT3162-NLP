# FIT3162-NLP Authorship Attribution Using Machine Learning and Natural Language Processing

I'm a Final Year Get Me Out of Here

- Mark Patricio
- Amelia Rowe

## How to install

- Python 3.x must be used
- If using system with Python 2.x and Python 3.x use `python3` command instead of `python`

### Installing Sci-kit Learn and NLTK

Go to your commandline/terminal and type the following command:

```bash
pip install -U numpy scipy scikit-learn nltk pytest
```

If you are using a unix system, use `pip3` command

If `pip` is not installed no your unix system, type the following into your terminal:

- Ubuntu/Debian:

```debian
sudo apt-get install python3-pip
```

- Fedora:

```fedora
dnf install python3-pip
```

## How to test it's installed properly

Go to Python interpreter and type:

```python
import numpy, scipy, nltk, sklearn
```

If no errors occur, then it's installed correctly.

## How to run the program

1. Check `config.py` for options to fine-grain classifier options

2. Run the `extract_features.py` file first to initially extract features from texts, arguments is english, spanish or both. This may take a while (10 minutes).

    ```bash
    python extract_features.py english|spanish|both
    ```

3. Run the `create_classifier.py` file to build the classifier, arguments is english, spanish or both

    ```bash
    python create_classifier.py english|spanish|both
    ```

4. Run the `main.py` file with text file as argument to be processed, as well as language

    ```bash
    python main.py file.txt english|spanish
    ```

## How to test the project ##

Navigate to project folder and type `pytest` in your terminal/command line.

## File and folder structure

### classifier folder

Folder containing python files which contain the classifier used to make predictions using Sci-kit Learn.

#### `english_classifier.py`

Python file containing the function used to construct the classifier which was found to be most accurate for our purposes and data set.

#### `number_classifier_test.py`

Python file which contains a test machine learning classifier, used to learn and understand how sci-kit learn works. Code adapted from a tutorial by sentdex.

### extraction folder

Folder containing files used for extracting items.

#### `feature_extraction.py`

Python file containing functions used to extract the features from texts, makes use of NLTK.

#### `file_extraction.py`

Python file containing functions used to extract strings out of files, makes use of Glob.

### original_texts folder

Folder containing the texts to be processed to be used for feature extraction.

#### Spanish Folder

Folder containing many folders of Spanish authors, with each author containing texts downloaded from Gutenberg project.

#### English Folder

Folder containing many folders of English authors, with each author containing texts downloaded from Gutenberg project.

### processed_texts folder

Folder containing the texts processed to be used for feature extraction.

#### Spanish Folder

Folder containing many folders of Spanish authors, with each author containing texts downloaded from Gutenberg project and processed.

#### English Folder

Folder containing many folders of English authors, with each author containing texts downloaded from Gutenberg project and processed.

#### `preprocessor.py`

Python file which given a folder name within that current directory will strip all unnecessary items within a Gutenberg project text and add the name of the directory to the top of the text.

### serialised_objects folder

Folder containing serialised objects (ie. variables saved as files) which can be loaded again without having to re-extract any other features

### experiment_texts folder

Test texts used to test the accuracy of the classifier as well as testing if the functions work as intended. These texts are not part of the corpus and are considered out of the training data.

### experiment_results folder

Folder containing the logs and results of testing from test_classifier.py, but also other manually tested items. This just contains raw data automaticallly/manually generated.

### `config.py`

Python file that contains many variables that are used in other python files within the root directory. Mostly contains constants, file paths and feature extraction settings. Changes here will require rerunning all other files for proper accuracy of other files.

### `create_classifier.py`

Python file used to create and train the classifier from the features extracted within `extract_features.py`. This includes loading serialised objects, reshaping the data and normalization before training and saving/serialising.

### `extract_features.py`

Python file which uses functions found in extraction function, to extract all text files within the processed_texts folder, process them, and extract all the necessary features before saving/serialising them.

### guitest.py

Python file temporarily created to test how the tkinter GUI library is used and how we can use it to create our own graphical user interface.

### main.py

Python file which is currently standing in place of a GUI which lets a user input any Gutenberg text, extract the features, load the classifier, and then makes a prediction on what author it is from.

### experiment_classifier.py

Python file used to test the accuracy of classifiers and features. Combinatorically tests all the combinations of features and and tests them with all the classifiers to test which classifier is the most accurate for our testing purposes.

### experiment_best_english_classifier.py

Python file used to test the accuracy of the best english classifiers and features. Combinatorically tests all the combinations of features and and tests them with all the classifiers to test which classifier is the most accurate for our testing purposes.

### best_english_classifiers folder

Folder directory containing all experiments in finding the best english classifiers and their settings