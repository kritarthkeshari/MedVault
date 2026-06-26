import pandas as pd

dataset = pd.read_csv("dataset.csv")

symptoms = set()

for column in dataset.columns:

    if column != "Disease":

        for symptom in dataset[column]:

            if pd.notna(symptom):

                symptoms.add(symptom.strip())

print("Total Symptoms:")
print(len(symptoms))

print("\nFirst 20 Symptoms:")

for symptom in sorted(list(symptoms))[:20]:

    print(symptom)