from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Training data
texts = [
    "Win money now",
    "Limited offer just for you",
    "Hey are we meeting today",
    "Don't forget the class tomorrow",
    "Congratulations you won a prize",
    "Can you send the notes"
]

# Labels: 1 = spam, 0 = not spam
labels = [1, 1, 0, 0, 1, 0]

# Test data
test_texts = [
    "Win cash prize now",
    "Are you coming to college today",
    "Free offer just for you",
    "Please send the assignment"
]

# True labels
true_labels = [1, 0, 1, 0]

# Convert text to numbers
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)
X_test = vectorizer.transform(test_texts)

# Train model
model = MultinomialNB()
model.fit(X_train, labels)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
