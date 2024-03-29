# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xrikK1bYVP-a1Xw9ByzbRhKcY1g5_35t
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import time

# Load the data
df = pd.read_csv("/content/drive/MyDrive/train.csv")

# Preprocess data
df['Category'] = df['Category'].str.lower()
df['Descript'] = df['Descript'].str.lower()

# Train-test split
train, test = train_test_split(df, test_size=0.3, random_state=60)

# Define the pipeline for Naive Bayes with Count Vectorizer and TF-IDF
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=['http', 'amp', 'rt', 't', 'c', 'the'])),
    ('classifier', MultinomialNB())
])

# Train the model
start_time = time.time()
nb_model = nb_pipeline.fit(train['Descript'], train['Category'])
end_time = time.time()
elapsed_time_training = end_time - start_time

# Make predictions
start_time = time.time()
nb_predictions = nb_model.predict(test['Descript'])
end_time = time.time()
elapsed_time_prediction = end_time - start_time

# Evaluate the model
accuracy = metrics.accuracy_score(test['Category'], nb_predictions)
classification_report = metrics.classification_report(test['Category'], nb_predictions)

# Print results
print('----------------------------- Naive Bayes Model -----------------------------')
print(' ')
print(f"Training Time: {elapsed_time_training:.2f} seconds")
print(f"Prediction Time: {elapsed_time_prediction:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(' ')
print('Classification Report:')
print(classification_report)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'vectorizer__max_features': [5000, 10000, 15000],
    'vectorizer__min_df': [1, 3, 5],
    'vectorizer__max_df': [0.8, 0.9, 1.0],
    'classifier__alpha': [0.1, 0.5, 1.0],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_train['Descript'], y_train)

# Print the best parameters and corresponding accuracy
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# Make predictions on the test data using the best model
y_pred = grid_search.predict(x_test['Descript'])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Data: ", accuracy)
