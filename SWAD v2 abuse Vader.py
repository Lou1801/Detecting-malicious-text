# Vader uses bag of words approach
import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("../SWAD v2.tsv", sep ='\t', header = None)

df = df.sample(frac=1).reset_index(drop=True)

abuse_data_1 = df[df.iloc[:, 2] == 'Yes'].head(100)

abuse_data_2 = df[df.iloc[:, 2] == 'No'].head(100)

combined_abuse_data = pd.concat([abuse_data_1, abuse_data_2]).reset_index(drop=True)

all_combined_val = []
all_words = []
for value1, value2 in zip(combined_abuse_data.iloc[:, 1], combined_abuse_data.iloc[:, 2]):
    # Process each value in the first column
    b_tags = re.compile(r'<\/?b>')
    words1 = b_tags.sub('', value1)

    punctuation = re.compile(r'[,()<>@#|0-9$%&*’+=\[\]{}^_\'"\\/]')
    words1 = punctuation.sub("", words1)

    words1 = word_tokenize(words1.lower())
    words1 = [word for word in words1 if word.lower() not in stopwords.words('english')]

    all_words.extend(words1)
    all_combined_val.append((words1, value2))
print(len(all_combined_val))

# Vader - Valence Aware Dictionary and Sentiment Reasoner, bag of words approach
# from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

all_combined_val1 = pd.DataFrame(all_combined_val, columns=['Words', 'Label'])
print(all_combined_val1.iloc[0])
# Run the polarity score on the entire dataset.
print(len(all_combined_val1))
sentiments = []
sentences = []
compound_scores = []
for index, row in all_combined_val1.iterrows():
    text = ' '.join(row['Words'])  # Join the list of words into a single string
    sentences.append(text)  # Append the sentence to the list of sentences

    compound_score = sia.polarity_scores(text)['compound']
    compound_scores.append(compound_score)

    if compound_score <= -0.05:
        sentiment = "Yes"
        sentiments.append(sentiment)

    else:
        sentiment = "No"
        sentiments.append(sentiment)

sentiments_df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiments, 'Compound Score': compound_scores})

relevant_labels = combined_abuse_data
# Concatenate the relevant labels with sentiments_df
sentiments_df = pd.concat([sentiments_df.reset_index(drop=True), relevant_labels.reset_index(drop=True).iloc[:, 2]], axis=1)

# Initialize variables to track correct predictions and total rows
correct_predictions = 0
total_rows = len(sentiments_df)
num_columns = sentiments_df.shape[1]

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
print("Accuracy:", f"{accuracy:.2f}%)")

