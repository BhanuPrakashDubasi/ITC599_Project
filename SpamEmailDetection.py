# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1yLls7QK0qMWRS8Ushbr_eDczY6dq9Zxw
"""

from google.colab import drive
drive.mount('/content/drive')

"""**Importing Dependecies**"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

"""**Data Collection & Pre-Processing**"""

df = pd.read_csv('/content/drive/MyDrive/project/mail_data.csv')
print(df)

# replace the null values with a null string
df = df.where((pd.notnull(df)),'')
df['Message'] = df['Message'].str.lower().str.replace(r'\W', ' ', regex=True)
df.head()

# Label Encoding : Encode labels to 0 (ham) and 1 (spam)
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])  # spam=1, ham=0

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

print(df)

"""**Splitting the data into training data & test data**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

"""**Building Logistic & Naive Bayes Models**"""

# Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

"""**Evaluation**"""

# confusion matrix
def plot_cm(cm, model_name):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/project/{model_name}_confusion_matrix.png')
    plt.close()

# Naive Bayes Metrics
nb_cm = confusion_matrix(y_test, nb_preds)
plot_cm(nb_cm, "NaiveBayes")
print("Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, nb_preds):.2f}")
print(f"Precision: {precision_score(y_test, nb_preds):.2f}")
print(f"Recall: {recall_score(y_test, nb_preds):.2f}")
print(f"F1 Score: {f1_score(y_test, nb_preds):.2f}")

# Logistic Regression Metrics
lr_cm = confusion_matrix(y_test, lr_preds)
plot_cm(lr_cm, "LogisticRegression")
print("\nLogistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
print(f"Precision: {precision_score(y_test, lr_preds):.2f}")
print(f"Recall: {recall_score(y_test, lr_preds):.2f}")
print(f"F1 Score: {f1_score(y_test, lr_preds):.2f}")

#Classification Report

print("=== Naive Bayes ===")
print(classification_report(y_test, nb_preds))
print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr_preds))

"""**Summarization**"""

from transformers import pipeline
import pandas as pd


# Use only the first 10 rows to keep it fast
sample_df = df[['Message']].dropna().head(10).copy()

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

# Generate summaries
summaries = []
for email in sample_df['Message']:
    try:
        summary = summarizer(email, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
    except:
        summary = "Summarization error"
    summaries.append(summary)

# Store and save results
sample_df['summary'] = summaries
sample_df.to_csv('/content/drive/MyDrive/project/email_summaries.csv', index=False)

print("✅ Summarization complete. Output saved to email_summaries.csv.")

"""**Reply Generation**"""

from transformers import pipeline
import pandas as pd

# Load sample emails
sample_df = df[['Message']].dropna().head(5).copy()

# Load text generation pipeline (GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Generate replies
replies = []
for email in sample_df['Message']:
    prompt = f"Reply to this email:\n\n{email}\n\nResponse:"
    try:
        response = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True)[0]['generated_text']
        # Remove the prompt from the generated output
        reply = response.replace(prompt, "").strip()
    except Exception as e:
        reply = "Reply generation error"
    replies.append(reply)

# Save results
sample_df['reply'] = replies
sample_df.to_csv('/content/drive/MyDrive/project/email_replies.csv', index=False)

print("✅ Reply generation complete. Output saved to email_replies.csv.")
