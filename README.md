# FIT3162-NLP
I'm a Final Year Get Me Out of Here

# TO INSTALL
Python 3.x must be used
If using system with Python 2.x and Python 3.x use `python3` command instead of `python`

## SCI-KIT LEARN and NLTK
```
pip install -U numpy scipy scikit-learn nltk
```

If you are using a unix system, use `pip3` command

If pip is not installed:
Ubuntu/Debian: `sudo apt-get install python3-pip`
Fedora: `dnf install python3-pip`

## Download Corpora and Data for NLTK
Go to Python interpreter (type `python` or `python3` in cmd/terminal) and type:

```
import nltk
download()
```

Then download all, or as necessary.

# TO TEST
Go to Python interpreter and type:

```
import numpy, scipy, nltk, sklearn
```

# TO RUN
0. Check `config.py` for options to fine-grain classifier options
   
1. Run the ClassifierProcess.py file first to initially train classifier, arguments is english, spanish or both

    ```
    python ClassifierProcess.py english|spanish|both
    ```

2. Run the main.py file with text file as argument to be processed, as well as language

    ```
    python main.py file.txt english|spanish
    ```
