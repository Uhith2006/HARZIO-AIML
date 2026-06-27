import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Sample facial features
data = {
    "FaceWidth": [14, 16, 13, 18, 15, 17],
    "FaceHeight": [20, 22, 19, 24, 21, 23],
    "EyeDistance": [6, 7, 5, 8, 6.5, 7.5],
    "Person": [
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
        "Bob"
    ]
}

df = pd.DataFrame(data)

X = df[["FaceWidth", "FaceHeight", "EyeDistance"]]
y = df["Person"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print("Face Recognition System")

width = float(input("Face Width: "))
height = float(input("Face Height: "))
eye = float(input("Eye Distance: "))

person = model.predict([[width, height, eye]])

print("Recognized Person:", person[0])