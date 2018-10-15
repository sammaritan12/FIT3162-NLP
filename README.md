# FIT3162-NLP Authorship Attribution Using Machine Learning and Natural Language Processing

I'm a Final Year Get Me Out of Here

- Mark Patricio
- Amelia Rowe

## How To Install

- Python 3.x must be used
- If using a unix system with Python 2.x and Python 3.x use `python3` command instead of `python`

### Installing Sci-kit Learn and NLTK

Go to your commandline/terminal and type the following command:

```bash
pip install -U numpy scipy scikit-learn nltk
```

If you are using a unix system, use `pip3` command

If `pip` is not installed on your unix system, type the following into your terminal:

- Ubuntu/Debian:

```debian
sudo apt-get install python3-pip
```

- Fedora:

```fedora
dnf install python3-pip
```

## How To Test it's Installed Properly

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

## How To Test The Project

Navigate to project folder and type the following in your terminal/command line:

```bash
python -m unittest
```

## File and folder structure

- **/classifier/** - fodler containing functions for classifiers
    - **english_classifier.py** - function to create english classifier
    - **spanish_classifier.py** - function to create spanish classifier
    - **test_english_classifier.py** - unit testing
    - **test_spanish_classifier.py** - unit testing
- **/experiment_results/** - text files containing some experiments done
- **/experiment_texts_out_of_sample/** - out of sample test data
    - **/bernard_shaw_test_text/** - texts by Bernard Shaw for testing
    - **/bernard_shaw_train_text/** - texts by Bernard Shaw for training
    - **/mark_twain_test_text/** - texts by Mark Twain for testing
- **extraction** - folder containing functions to extract files and features
    - **/test/** - unit testing
    - **feature_extraction.py** - extracts features from processed text
    - **file_extraction.py** - extracts text from files
    - **test_feature_extraction.py** - unit testing
    - **test_file_extraction.py** - unit testing
- **/original_texts/** - folder containing original Gutenberg texts
    - **/English/** - English texts from English authors
    - **/Spanish/** - Spanish texts from Spanish authors
- **/processed_texts/** - folder containing processed Gutenberg texts
    - **/english/** - English texts from English authors
    - **/spanish/** - Spanish texts from Spanish authors
    - **preprocessor.py** - function to preprocess a folder of Gutenberg texts
    - **test_preprocessor.py** - unit testing
- **/serialised_objects/** - folder containing saved files
- **config.py** - file to configure the whole projects
- **create_classifier.py** - creates classifier/s from serialised extracted features
- **experiment_best_english_classifier.py** - function to test the best english classifiers
- **experiment_classifier.py** - function to experiment which classifier to use
- **extract_features.py** - function to extract features from processed texts
- **finetune_classifier.py** - function to finetune certain classifiers for accuracy
- **main.py** - function to test any processed Gutenberg text
- **README.md** - This file
- **test_experiment_classifier.py** - unit testing
- **text_extract_features.py** - unit testing
