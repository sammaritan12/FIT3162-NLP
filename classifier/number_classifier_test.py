# accessed 06/08/2018
# https://www.youtube.com/watch?v=KTeVOb8gaD4
# followed tutorial to understand how sklearn works

# library used to show digits
import matplotlib.pyplot as plt

# library for dataset where digit is from and SVM classifier
from sklearn import datasets
from sklearn import svm

# test dataset of pictures of images
digits = datasets.load_digits()

# prints the data, an array of arrays, each with a numerical representation of a picture
# e.g. 0 is white, and 9 is the darkest black pixel
print(digits.data)
# for every data item, it has a correct classification, this is it
print(digits.target)

# this is the SVM classifier
# gamma affects the 'jumps' to gradient descent
# higher gamma = faster but less accurate
clf = svm.SVC(gamma=0.001, C=100)

# prints how many digits are there
print(len(digits.data))

# x is actual data set, y is what we want these data sets to be classified to
# we are choosing every number except last so we can predict this
x, y = digits.data[:-1], digits.target[:-1]

# fitting the data of the digits to actual targets
clf.fit(x, y)

# this is where the SVM predicts the next things, here it is the last item
print('Prediction:', clf.predict(digits.data[-1].reshape(1, -1)))

# just printing what the digit looks like, convert numbers to a pixel representation
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()