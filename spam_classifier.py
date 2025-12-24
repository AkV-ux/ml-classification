from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test message
test_message = ["Win a free phone now"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")
