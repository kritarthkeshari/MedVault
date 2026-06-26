import pandas as pd

dataset = pd.read_csv("dataset.csv")

# Build symptom vocabulary
symptoms = set()

for column in dataset.columns:

    if column != "Disease":

        for symptom in dataset[column]:

            if pd.notna(symptom):

                symptoms.add(
                    symptom.strip().lower()
                )

def extract_symptoms(text):

    text = text.lower()

    found = []

    for symptom in symptoms:

        symptom_text = symptom.replace("_", " ")

        if symptom_text in text:

            found.append(symptom)

    return found


sample_text = """
Patient has fever, cough and headache
for the last 3 days.
"""

print(extract_symptoms(sample_text))