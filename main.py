import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv")

print("Training Data:\n")
print(data)

# Attributes (all columns except target)
concepts = data.iloc[:, :-1].values

# Target column
target = data.iloc[:, -1].values

# Initialize hypothesis
hypothesis = None

print("\nApplying FIND-S Algorithm...\n")

for i in range(len(concepts)):

    if target[i] == "Yes":

        if hypothesis is None:
            hypothesis = concepts[i].copy()

        else:

            for j in range(len(hypothesis)):

                if hypothesis[j] != concepts[i][j]:
                    hypothesis[j] = "?"

        print("After Positive Example", i + 1)
        print(hypothesis)
        print()

print("----------------------------------")
print("Final Most Specific Hypothesis")
print("----------------------------------")
print(hypothesis)