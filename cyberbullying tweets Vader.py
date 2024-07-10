# Vader uses bag of words approach

import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("../cyberbullying_tweets.csv", sep=',')

abuse_data_1 = df[df['cyberbullying_type'] == 'age'].head(1250)

abuse_data_2 = df[df['cyberbullying_type'] == 'ethnicity'].head(1250)

combined_abuse_data = pd.concat([abuse_data_1, abuse_data_2])
combined_abuse_data['cyberbullying_type'] = 1

non_abusive_rows = df[df['cyberbullying_type'] == 'not_cyberbullying'].head(2500)
non_abusive_rows['cyberbullying_type'] = 0

complete_combined_data = pd.concat([combined_abuse_data, non_abusive_rows]).reset_index(drop=True)

all_combined_val = []
all_words = []
for value1, value2 in zip(complete_combined_data.iloc[:, 0], complete_combined_data.iloc[:, 1]):
    # Process each value in the first column
    b_tags = re.compile(r'<\/?b>')
    words1 = b_tags.sub('', value1)

    punctuation = re.compile(r'[,()<>@#|0-9$%&*’+=\[\]{}^_\'"\\/]')
    words1 = punctuation.sub("", words1)

    words1 = word_tokenize(words1.lower())
    words1 = [word for word in words1 if word.lower() not in stopwords.words('english')]

    all_words.extend(words1)
    all_combined_val.append((words1, value2))

# Vader - Valence Aware Dictionary and Sentiment Reasoner, bag of words approach

sia = SentimentIntensityAnalyzer()

all_combined_val1 = pd.DataFrame(all_combined_val, columns=['Words', 'Label'])
# Run the polarity score on the entire dataset.
sentiments = []
sentences = []
compound_scores = []
for index, row in all_combined_val1.iterrows():
    text = ' '.join(row['Words'])  # Join the list of words into a single string
    sentences.append(text)  # Append the sentence to the list of sentences

    compound_score = sia.polarity_scores(text)['compound']
    compound_scores.append(compound_score)

    if compound_score <= -0.05:
        sentiment = 1
        sentiments.append(sentiment)

    else:
        sentiment = 0
        sentiments.append(sentiment)

sentiments_df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiments, 'Compound Score': compound_scores})

relevant_labels = complete_combined_data
# Concatenate the relevant labels with sentiments_df
sentiments_df = pd.concat([sentiments_df.reset_index(drop=True), relevant_labels.reset_index(drop=True).iloc[:, 1]], axis=1)

# Initialize variables to track correct predictions and total rows
correct_predictions = 0
total_rows = len(sentiments_df)

# Iterate over each row in the DataFrame
for index, row in sentiments_df.iterrows():
    # Get the predicted sentiment and true sentiment label
    predicted_sentiment = row.iloc[1]
    true_sentiment = row.iloc[3]

    # Compare the predicted sentiment with the true sentiment label
    if predicted_sentiment == true_sentiment:
        correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_rows) * 100
print("Accuracy:", accuracy)

