# Vader uses bag of words approach
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

plt.style.use('ggplot')

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

data_file = pd.read_excel('../ICWSM18 - SALMINEN ET AL.xlsx', sheet_name="All comments")

data_file['sentiment'] = 0
for index, row in data_file.iterrows():
    if row['Class'] == 'Neutral':
        data_file.at[index,'sentiment'] = 0
    else:
        data_file.at[index,'sentiment'] = 1

df_shuffled = data_file.sample(frac=1)

abuse_data = df_shuffled[df_shuffled['sentiment'] == 1].head(500)

non_abusive_rows = df_shuffled[df_shuffled['sentiment'] == 0].head(500)

complete_combined_data = pd.concat([abuse_data, non_abusive_rows]).reset_index(drop=True)

all_combined_val = []
all_words = []
for value1, value2 in zip(complete_combined_data.iloc[:, 3], complete_combined_data.iloc[:, 10]):
    # Process each value in the first column

    punctuation = re.compile(r'[-.,:;()<>@#|0-9$%&*’+=\[\]{}^_\'"\\/]')
    words1 = punctuation.sub("", value1)

    words1 = word_tokenize(words1.lower())
    words1 = [word for word in words1 if word.lower() not in stopwords.words('english')]

    all_words.extend(words1)
    all_combined_val.append((words1, value2))

# Vader - Valence Aware Dictionary and Sentiment Reasoner, bag of words approach


def polarity_scores_roberta(example):
    # puts it into embeddings that the model can understand
    # print(encoded_text)
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    # print(output)
    scores = output[0][0].detach().numpy()
    # print(scores)
    scores = softmax(scores)
    # print(scores)
    # softmax shows neg, neu, and pos

    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


sia = SentimentIntensityAnalyzer()


def calculate_compound_score(roberta_scores):
    weights = {'roberta_pos': 0.5, 'roberta_neg': -0.5, 'roberta_neu': 0.1}

    sum_of_weights = sum(roberta_scores[sentiment] * weights[sentiment] for sentiment in roberta_scores)

    roberta_compound_score = sum_of_weights / len(roberta_scores)
    return roberta_compound_score


all_combined_val1 = pd.DataFrame(all_combined_val, columns=['Words', 'Label'])
# Run the polarity score on the entire dataset.
ratings = {}
vader_sentiments = []
roberta_sentiments = []
sentences = []
vader_compound_scores = []
roberta_compound_scores = []

for index, row in all_combined_val1.iterrows():
    text = ' '.join(row['Words'])  # Join the list of words into a single string
    sentences.append(text)  # Append the sentence to the list of sentences

    vader_result = sia.polarity_scores(text)
    vader_result_rename = {}
    for rating, value in vader_result.items():
        vader_result_rename[f"vader_{rating}"] = value

    roberta_result = polarity_scores_roberta(text)
    both = {**vader_result_rename, **roberta_result}
    ratings[index] = both

    roberta_compound = calculate_compound_score(roberta_result)
    roberta_compound_scores.append(roberta_compound)

    compound_score = ratings[index]['vader_compound']
    vader_compound_scores.append(compound_score)

    if compound_score <= -0.05:
        sentiment = 1
        vader_sentiments.append(sentiment)

    else:
        sentiment = 0
        vader_sentiments.append(sentiment)

    if roberta_compound <= -0.05:
        sentiment = 1
        roberta_sentiments.append(sentiment)

    else:
        sentiment = 0
        roberta_sentiments.append(sentiment)


sentiments_df = pd.DataFrame({'sentence': sentences, 'vader_sentiment': vader_sentiments, 'roberta_compound': roberta_compound_scores, 'roberta_sentiments': roberta_sentiments})

# Select the first 500 relevant labels associated with abuse_data_1 and abuse_data_2
relevant_labels = pd.concat([complete_combined_data])
# Concatenate the relevant labels with sentiments_df
sentiments_df = pd.concat([sentiments_df.reset_index(drop=True), relevant_labels.reset_index(drop=True).iloc[:, 10]], axis=1)

vader_correct_predictions = 0
roberta_correct_predictions = 0
total_rows = len(sentiments_df)


# Iterate over each row in the DataFrame
for index, row in sentiments_df.iterrows():
    # Get the predicted sentiment and true sentiment label
    vader_predicted_sentiment = row.iloc[1]
    true_sentiment = row.iloc[4]
    roberta_predicted_sentiment = row.iloc[3]

    # Compare the predicted sentiment with the true sentiment label
    if vader_predicted_sentiment == true_sentiment:
        vader_correct_predictions += 1

    if roberta_predicted_sentiment == true_sentiment:
        roberta_correct_predictions += 1

# Calculate accuracy
vader_accuracy = (vader_correct_predictions / total_rows) * 100
print("Vader Accuracy:", vader_accuracy)

roberta_accuracy = (roberta_correct_predictions / total_rows) * 100
print("ROBERTA Accuracy:", roberta_accuracy)

results_df = pd.DataFrame(ratings).T
results_df = pd.concat([results_df.reset_index(drop=True), sentiments_df.reset_index(drop=True)], axis=1)

sns.pairplot(data = results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos', 'vader_compound', 'roberta_compound'],
             hue = results_df.columns[11],
             palette = 'tab10',
             height = 2,
             aspect = 1)

sns.pairplot(data = results_df, vars=['vader_compound', 'roberta_compound'],
             hue = results_df.columns[11],
             palette = 'tab10',
             height = 2,
             aspect = 1)
plt.show()

for column_header in results_df.columns:
    print(f"{column_header}: {results_df[column_header].iloc[0]}")


