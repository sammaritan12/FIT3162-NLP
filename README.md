# FIT3162-NLP Authorship Attribution Using Machine Learning and Natural Language Processing

I'm a Final Year Get Me Out of Here

- Mark Patricio
- Amelia Rowe

## TO INSTALL

- Python 3.x must be used
- If using system with Python 2.x and Python 3.x use `python3` command instead of `python`

### SCI-KIT LEARN and NLTK

```bash
pip install -U numpy scipy scikit-learn nltk
```

If you are using a unix system, use `pip3` command

If `pip` is not installed no your unix system:

- Ubuntu/Debian:

```debian
sudo apt-get install python3-pip
```

- Fedora:

```fedora
dnf install python3-pip
```

### Optional: Download Corpora and Data for NLTK

Go to Python interpreter (type `python` or `python3` in cmd/terminal) and type:

```python
import nltk
download()
```

Then download all, or as necessary.

## Testing imports

Go to Python interpreter and type:

```python
import numpy, scipy, nltk, sklearn
```

## Running the program

1. Check `config.py` for options to fine-grain classifier options

2. Run the `extract_features.py` file first to initially extract features from texts, arguments is english, spanish or both. This may take a while.

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
