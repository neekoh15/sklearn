from sklearn.datasets import load_files

# Load the IMDb movie reviews dataset
movie_reviews = load_files('path/to/imdb_dataset', shuffle=True)

X = movie_reviews.data  # Text data
y = movie_reviews.target  # Sentiment labels (0 for negative, 1 for positive)

# Step 2: Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Splitting the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Model Selection (Multinomial Naive Bayes)
from sklearn.naive_bayes import MultinomialNB

# Create a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Step 5: Model Training
clf.fit(X_train, y_train)

# Step 6: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Predictions
new_reviews = ["This movie was amazing! I loved every moment of it.", "Terrible film, waste of time."]
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)
new_predictions = clf.predict(new_reviews_tfidf)

for review, prediction in zip(new_reviews, new_predictions):
    sentiment = "positive" if prediction == 1 else "negative"
    print(f"Review: '{review}'\nPredicted Sentiment: {sentiment}\n")
