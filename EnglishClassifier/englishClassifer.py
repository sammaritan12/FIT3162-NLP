from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split

# TODO vectoriser for texts, turns text to numbers
# this one vectorises 1grams
def textVectorizer(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)


text = textVectorizer(['the quick brown', 'fox jumps over', 'the lazy dog', 'quick brown'])
svm = LinearSVC()
svm.fit(text[:-1], ['A','B','C']) # tries to fit text with a,b,c using machine learning
print(svm.predict(text[-1]))