
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
np.random.seed(60)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("/content/train.csv")

df.info()

df.head()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')

# Define extra stopwords
extra_stopwords = ['http', 'amp', 'rt', 't', 'c', 'the']

# Tokenizer using NLTK
tokenizer = RegexpTokenizer(r'\w+')

stop_words = list(stopwords.words('english')) + list(extra_stopwords)

# Custom transformer using NLTK's RegexpTokenizer
class NLTKTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join(self.tokenizer.tokenize(text)) for text in X]

# Using TF-IDF to vectorize features instead of CountVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000, min_df=5)

classifier = MultinomialNB()

a = df.drop("Category", axis=1)
b = df["Category"]
x_train , x_test ,y_train, y_test = train_test_split(a,b, test_size=0.3 , random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tokenizer', NLTKTokenizer()),
    ('vectorizer', count_vectorizer),  # or tfidf_vectorizer
    ('classifier', classifier)
])

print("Number of samples in X_train:", len(x_train))
print("Number of samples in y_train:", len(y_train))

print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

pipeline.fit(x_train['Descript'], y_train)

import time

start_time = time.time()
pipeline.fit(x_train['Descript'], y_train)


# Make predictions on the test data
y_pred = pipeline.predict(x_test['Descript'])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
# Print execution time and accuracy
print(f"Execution Time: {end_time - start_time} seconds")
print(f"Accuracy: {accuracy}")

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

