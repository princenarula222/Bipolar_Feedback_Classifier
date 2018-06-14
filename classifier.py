import csv
import numpy as np
import re
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.metrics import f1_score
import random


f = open('train.csv', 'r')
train = list(csv.reader(f, delimiter=','))
random.shuffle(train)
n = 200000                                                         # number of training examples
train = train[0:n]
train = np.array(train)
label = train[:, 0]
label = label.astype(int)
train = train[:, 1]
label = (label+1) % 2                                              # label=0 : negative class, label=1 : positive class

f = open('test.csv', 'r')
test = list(csv.reader(f, delimiter=','))
m = 20000                                                          # number of testing examples
test = test[0:m]
test = np.array(test)
y = test[:, 0]
y = y.astype(int)
test = test[:, 1]
y = (y+1) % 2                                                      # y=0: negative class, y=1: positive class


def preprocess(h):
    h = h.lower()
    h = h.replace('\n', '').replace('\r', '')
    h = re.sub(r'[^\w\d\s]+', '', h)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    h = pattern.sub('', h)
    h = h.split()
    return h


filename = 'GoogleNews-vectors-negative300.bin'                              # Using google's word embedding model
model = KeyedVectors.load_word2vec_format(filename, binary=True)


x = np.zeros((n, 300))
x_test = np.zeros((m, 300))


for i in range(n):
    t = preprocess(train[i])
    for j in t:
        if j not in model:
            continue
        x[i] = np.add(x[i], model[j])


for i in range(m):
    t = preprocess(test[i])
    for j in t:
        if j not in model:
            continue
        x_test[i] = np.add(x_test[i], model[j])


clf = svm.SVC()
clf.fit(x, label)                                                              # using a support vector machine
y_pred = clf.predict(x_test)
diff = np.subtract(y, y_pred)
print(diff)
print(f1_score(y, y_pred))


y = y.reshape((m, 1))
y_pred = y_pred.reshape((m, 1))
diff = diff.reshape((m, 1))
b = open('label_test.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(y)
b.close()

b = open('predicted_test.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(y_pred)
b.close()

b = open('difference.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(diff)
b.close()
