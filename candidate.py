import pandas as pd
data = pd.read_csv("candidate.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values
S = concepts[0].copy()
G = [["?" for _ in range(len(S))]]
for i in range(len(target)):
    if target[i] == "Yes":
        for j in range(len(S)):
            if concepts[i][j] != S[j]:
                S[j] = "?"
    else:
        for j in range(len(S)):
            if concepts[i][j] != S[j]:
                G.append(["?" if k != j else S[j] for k in range(len(S))])
print("Specific Hypothesis:")
print(S)
print("\nGeneral Hypothesis:")
for g in G:
    print(g)