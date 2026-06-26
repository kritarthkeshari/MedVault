import pandas as pd

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

symptom_list = sorted(list(symptoms))


def extract_symptoms(text):

    text = text.lower()

    found = []

    for symptom in symptom_list:

        symptom_text = symptom.replace(
            "_",
            " "
        )

        if symptom_text in text:

            found.append(symptom)

    return found


def create_feature_vector(found_symptoms):

    vector = []

    for symptom in symptom_list:

        if symptom in found_symptoms:

            vector.append(1)

        else:

            vector.append(0)

    return vector


text = """
Patient has fever and cough
"""

found = extract_symptoms(text)

vector = create_feature_vector(found)

print("Symptoms Found:")
print(found)

print("\nVector Length:")
print(len(vector))

print("\nFirst 20 Values:")
print(vector[:20])