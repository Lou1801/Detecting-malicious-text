import nltk
import random
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import KFold

from sklearn.naive_bayes import  MultinomialNB, GaussianNB, BernoulliNB
# can change parameters of these

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


pd.set_option('max_colwidth', 2000)
df = pd.read_csv("../SWAD v2.tsv", sep ='\t', header = None)
test = df.iloc[0, 2]
# Iterate through the first column (index 0)

abuse_data_1 = df[df.iloc[:, 2] == 'Yes'].head(1250)

abuse_data_2 = df[df.iloc[:, 2] == 'No'].head(1250)

combined_abuse_data = pd.concat([abuse_data_1, abuse_data_2])

# print(combined_abuse_data)

all_combined_val = []
all_words = []
for value1, value2 in zip(combined_abuse_data.iloc[:, 1], combined_abuse_data.iloc[:, 2]):
    # Process each value in the first column
    b_tags = re.compile(r'<\/?b>')
    words1 = b_tags.sub('', value1)

    punctuation = re.compile(r'[-.?!,:;()<>@#|0-9$%&*’+=\[\]{}^_'"'/]")
    words1 = punctuation.sub("", words1)

    words1 = word_tokenize(words1.lower())
    words1 = [word for word in words1 if word.lower() not in stopwords.words('english')]

    all_words.extend(words1)
    all_combined_val.append((words1, value2))


all_words_freq = nltk.FreqDist(all_words)
# ordered from most common to least common, nltk's own dictionary of sorts
print(all_words_freq.most_common(15))

word_features = list(all_words_freq.keys()) [:]
# keys is a method of dictionaries to get all the values, in this case it gets all the words.
#print(word_features)


def find_features(sentence):
    # Convert the document (list of words) into a set to remove duplicates
    # document is one movie review entry
    words = set(sentence)
    # print(words)

    # Initialize an empty dictionary to store the features
    features = {}
    # Iterate over each word feature in the word_features list
    for w in word_features:
        # Check if the current word feature is present in the set of words
        # If the word feature is present, assign True to the feature
        # If the word feature is not present, assign False to the feature
        features[w] = (w in words)
        # print(w)
    # Return the dictionary containing the features
    return features


# print(featuresets)
i = 0
for count in all_words:
    i += 1
# print(i)
# print(all words)

# Prepare data for cross-validation
k = 5
X = all_combined_val  # Features
random.shuffle(X)
y = [label for _, label in X]  # Target variable, the _ means that we ignore the first column

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation
scores = []
for train_index, test_index in kf.split(X):
    print("Train indices:", len(train_index))
    print("Test indices:", len(test_index))
    X_train = [X[i] for i in train_index]
    X_test = [X[i] for i in test_index]

    training_set = [(find_features(words), label) for words, label in X_train]
    testing_set = [(find_features(words), label) for words, label in X_test]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(classifier.show_most_informative_features())
    accuracy = nltk.classify.accuracy(classifier, testing_set)

    scores.append(accuracy)

# Print the accuracy scores for each fold
for i, score in enumerate(scores):
    percentage = score * 100
    print(f"Fold {i+1} accuracy: {percentage:.2f}%")

# Calculate and print the average accuracy across all folds
average_accuracy = sum(scores) / len(scores)
average_percentage = average_accuracy * 100
print("Average Accuracy:", f"{average_percentage:.2f}%")
print(len(word_features))
print(len(all_combined_val))

# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Naive Bayes algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
# most_informative_features = classifier.show_most_informative_features(500)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB Naive Bayes algo accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
#
# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print("Bernoulli Naive Bayes algo accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
#
# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier algo accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
#
# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier algo accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)
#
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier algo accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)
#
# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print(" LinearSVC_classifier algo accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
#
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier algo accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
#

