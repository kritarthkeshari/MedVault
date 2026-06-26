import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("dataset.csv")

# Build master symptom list
symptoms = set()

for column in dataset.columns:

    if column != "Disease":

        for symptom in dataset[column]:

            if pd.notna(symptom):

                symptoms.add(
                    symptom.strip().lower()
                )

symptom_list = sorted(
    list(symptoms)
)

X = []

for _, row in dataset.iterrows():

    vector = []

    row_symptoms = []

    for column in dataset.columns:

        if column != "Disease":

            if pd.notna(row[column]):

                row_symptoms.append(
                    row[column].strip().lower()
                )

    for symptom in symptom_list:

        if symptom in row_symptoms:

            vector.append(1)

        else:

            vector.append(0)

    X.append(vector)

X = np.array(X)

y = dataset["Disease"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("X Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Complete")
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)
def predict_disease(text):

    text = text.lower()

    found_symptoms = []

    for symptom in symptom_list:

        symptom_text = symptom.replace(
            "_",
            " "
        )

        if symptom_text in text:

            found_symptoms.append(symptom)

    vector = []

    for symptom in symptom_list:

        if symptom in found_symptoms:

            vector.append(1)

        else:

            vector.append(0)

    prediction = model.predict([vector])

    disease = encoder.inverse_transform(
        prediction
    )

    return disease[0]
print(
    predict_disease(
        "loss of balance lack of concentration"
    )
)