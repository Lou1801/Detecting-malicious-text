import nltk
import random
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# can change parameters of these

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

data_file = pd.read_excel('../ICWSM18 - SALMINEN ET AL.xlsx', sheet_name="All comments")

data_file['sentiment'] = 0
for index, row in data_file.iterrows():
    if row['Class'] == 'Neutral':
        data_file.at[index,'sentiment'] = 0
    else:
        data_file.at[index,'sentiment'] = 1

# print(data_file)

abuse_data = data_file[data_file['sentiment'] == 1].head(500)

non_abusive_rows = data_file[data_file['sentiment'] == 0].head(500)

complete_combined_data = pd.concat([abuse_data, non_abusive_rows])
print(len(complete_combined_data))
all_combined_val = []
all_words = []
for value1, value2 in zip(complete_combined_data.iloc[:, 3], complete_combined_data.iloc[:, 10]):
    # Process each value in the first column

    punctuation = re.compile(r'[-.?!,:;()<>@#|0-9$%&*’+=\[\]{}^_'"'/]")
    words1 = punctuation.sub("", value1)

    words1 = word_tokenize(words1.lower())
    words1 = [word for word in words1 if word.lower() not in stopwords.words('english')]

    all_words.extend(words1)
    all_combined_val.append((words1, value2))


all_words_freq = nltk.FreqDist(all_words)
# ordered from most common to least common, nltk's own dictionary of sorts
# print(all_words_freq.most_common(15))

word_features = list(all_words_freq.keys())[:]
# keys is a method of dictionaries to get all the values, in this case it gets all the words.
# print(word_features)


def find_features(sentence):
    # Convert the document (list of words) into a set to remove duplicates
    # document is one movie review entry
    words = set(sentence)
    #print(words)


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


i = 0
for count in all_words:
    i += 1
# print(i)
# print(all words)

# training_set = featuresets[:1500]
# testing_set = featuresets[1500:]

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

    # Prepare training and testing sets
    training_set = [(find_features(words), label) for words, label in X_train]

    testing_set = [(find_features(words), label) for words, label in X_test]
    # print(testing_set)
    # Train the classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(classifier.show_most_informative_features())
    # Evaluate the classifier
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


# abuse_data_1_length = df[df['cyberbullying_type'] == 'other_cyberbullying']
# total_rows_matched = len(abuse_data_1_length)
# print("Total rows matched:", total_rows_matched)

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


