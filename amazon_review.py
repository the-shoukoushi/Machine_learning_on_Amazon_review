import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/shoukoushi/Downloads/amazon_alexa.tsv')
dataset

dataset.info()
dataset.describe()
dataset['verified_reviews']
dataset = dataset.drop(['date'], axis=1).drop(['variation'], axis=1)
dataset

sns.countplot(dataset['rating'], label = 'count')

sns.countplot(dataset['feedback'], label= 'count')

dataset['length'] = dataset['verified_reviews'].apply(len)
dataset

positive_reviews = dataset[dataset['feedback']==1]
positive_reviews

negative_reviews = dataset[dataset['feedback']==0]
negative_reviews

reviews=dataset['verified_reviews'].tolist()
len(reviews)

combined_reviews=' '.join(reviews)
combined_reviews

import string
string.punctuation

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

def message_cleaning(message):
    punctuation_removed = [char for char in message if char not in string.punctuation]
    punctuation_removed_join = ''.join(punctuation_removed)
    punctuation_removed_join_clean = [word for word in punctuation_removed_join.split() if word.lower() not in stopwords.words('english')]
    return punctuation_removed_join_clean

dataset_clean = dataset['verified_reviews'].apply(message_cleaning)
print(dataset_clean [2])
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
reviews_countvectorizer = vectorizer.fit_transform(dataset['verified_reviews'])
print(vectorizer.get_feature_names())

print(reviews_countvectorizer.toarray())

A = pd.DataFrame(reviews_countvectorizer.toarray())
A

B = dataset['feedback']
def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');
A.shape
B.shape

from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(A_train, B_train)

from sklearn.metrics import classification_report

yhat_NB = NB_classifier.predict(A_test)
plot_confusion_matrix(B_test, yhat_NB)

print (classification_report(B_test, yhat_NB))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(A_train, B_train)

yhat_logreg = logreg.predict(A_test)
plot_confusion_matrix(B_test, yhat_logreg)

print(classification_report(B_test, yhat_logreg))
