from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# Training data
texts = [
    # Spam
    "Win money now",
    "Limited offer just for you",
    "Congratulations you won a prize",
    "Free cash offer",
    "Win a free phone",
    "Exclusive deal just for you",
    "Claim your reward now",
    "Earn money fast",

    # Not spam
    "Hey are we meeting today",
    "Don't forget the class tomorrow",
    "Can you send the notes",
    "Please send the assignment",
    "Are you coming to college today",
    "Let's meet after class",
    "Can we reschedule the meeting",
    "Did you complete the homework"
]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1,   # spam
    0, 0, 0, 0, 0, 0, 0, 0    # not spam
]

# Test data (trickier messages)
test_texts = [
    "Meeting offer today",
    "Free class notes",
    "Win meeting prize",
    "Are you free now"
]

# True labels
true_labels = [0, 0, 1, 0]


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
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)

print("Precision:", precision)
print("Recall:", recall)

