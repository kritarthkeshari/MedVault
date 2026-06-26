import pandas as pd

dataset = pd.read_csv("dataset.csv")
severity = pd.read_csv("Symptom-severity.csv")
description = pd.read_csv("symptom_Description.csv")
precaution = pd.read_csv("symptom_precaution.csv")

print("Disease Dataset Shape:")
print(dataset.shape)

print("\nSeverity Dataset Shape:")
print(severity.shape)

print("\nDescription Dataset Shape:")
print(description.shape)

print("\nPrecaution Dataset Shape:")
print(precaution.shape)