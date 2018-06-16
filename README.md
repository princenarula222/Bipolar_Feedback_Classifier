# Bipolar Feedback Classifier
This repository provides an implementation of a bipolar feedback classifier which segregates positive and negative customer reviews.
The implementation utilizes a support vector machine model on the Yelp review dataset for training and testing purposes.
Google's pretrained Word2Vec embedding model has been utilized for generating the word vectors using the Gensim module in Python.

Refer the following link for a better understanding of word embeddings in Python.

https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

# Dependencies
Package - Appropriate Python package

Frameworks - Numpy, NLTK, Gensim, Scikit-learn

# Getting started
Download the essential files using the following link:

https://drive.google.com/drive/folders/1uPKPqKDc6Ka5rLkEz6do6AzPwZs3TNp7?usp=sharing

Place these files in the root folder of the repository.

# Training and testing the model
Run 'classifier.py' to train and test the model.

# Result
Following files are generated in the root folder upon completion of training.

label_test.csv - stores true labels of the reviews contained in test set

predicted_test - stores predicted labels of the reviews contained in test set

difference.csv - stores the arithmetic difference between label_test.csv and predicted_test.csv

# Interpretation
y=0: Negative review 

y=1: Positive review

# Performance analysis
I have placed my results in 'result'(result/) folder for reference. 

No. of training examples used: 200000

No. of testing examples used: 20000

F1 Score: 0.8269542381404704
