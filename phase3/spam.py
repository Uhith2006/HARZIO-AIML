import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset
data = {
    "Email": [
        "Congratulations! You won a free iPhone",
        "Meeting is scheduled at 10 AM tomorrow",
        "Claim your free prize now",
        "Project report has been submitted",
        "Earn money quickly from home",
        "Can we have lunch tomorrow?",
        "Limited offer! Buy now and save 50%",
        "Please review the attached document",
        "Win cash instantly by clicking here",
        "Let's discuss the assignment today"
    ],
    "Label": [
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])
y = df["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test custom email
while True:
    email = input("\nEnter an email (or type 'exit' to quit): ")

    if email.lower() == "exit":
        print("Program Ended.")
        break

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)

    print("Prediction:", prediction[0])