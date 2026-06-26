# import pandas as pd

# df = pd.read_csv("dataset.csv")

# print(df)
# import pandas as pd

# df = pd.read_csv("dataset.csv")

# print(df)

# print("\nRows and Columns:")
# print(df.shape)
# import pandas as pd

# from biobert import get_embedding

# df = pd.read_csv("dataset.csv")

# first_patient = df.iloc[0]

# print(first_patient)

# embedding = get_embedding(
#     first_patient["symptoms"]
# )

# print("\nEmbedding Shape:")
# print(embedding.shape)
import pandas as pd
import numpy as np

from biobert import get_embedding
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")

# Create feature matrix X
X = []

for _, row in df.iterrows():

    embedding = get_embedding(
        row["symptoms"]
    )

    X.append(embedding)

X = np.array(X)

print("Feature Matrix Shape:")
print(X.shape)

# Create labels y
y = df["diagnosis"]

print("\nLabels:")
print(y)

# Encode labels
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nEncoded Labels:")
print(y_encoded)

# Show mapping
print("\nClass Mapping:")

for disease, code in zip(
    encoder.classes_,
    range(len(encoder.classes_))
):
    print(f"{disease} -> {code}")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

print("\nModel Created Successfully")
print(model)
model.fit(X, y_encoded)

print("\nModel Training Complete")
test_symptom = "tight chest pain sweating"

test_embedding = get_embedding(test_symptom)

prediction = model.predict(
    [test_embedding]
)

predicted_disease = encoder.inverse_transform(
    prediction
)

print("\nPrediction:")
print(predicted_disease[0])
