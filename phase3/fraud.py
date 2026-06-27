import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "Amount": [100, 5000, 200, 10000, 150, 8000, 250, 12000],
    "Location": [0, 1, 0, 1, 0, 1, 0, 1],  # 0=Known, 1=Unknown
    "TransactionsToday": [1, 10, 2, 15, 1, 12, 2, 18],
    "Fraud": [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["Amount", "Location", "TransactionsToday"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# User Input
amount = float(input("Transaction Amount: "))
location = int(input("Location (0=Known,1=Unknown): "))
transactions = int(input("Transactions Today: "))

result = model.predict([[amount, location, transactions]])

if result[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")