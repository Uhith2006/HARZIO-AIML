import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Fever": [1,1,0,1,0,0,1,0],
    "Cough": [1,0,1,1,0,1,1,0],
    "Fatigue": [1,1,0,1,0,0,1,0],
    "Disease": [1,1,0,1,0,0,1,0]
}

df = pd.DataFrame(data)

X = df[["Fever","Cough","Fatigue"]]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print("\nEnter Patient Details")

fever = int(input("Fever (1=Yes,0=No): "))
cough = int(input("Cough (1=Yes,0=No): "))
fatigue = int(input("Fatigue (1=Yes,0=No): "))

prediction = model.predict([[fever, cough, fatigue]])

if prediction[0] == 1:
    print("Disease Detected")
else:
    print("No Disease Detected")