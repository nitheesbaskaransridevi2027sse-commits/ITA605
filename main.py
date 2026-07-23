#Probabilities of selecting each box
P_A = 1 / 4
P_B = 2 / 4
P_C = 1 / 4

#Probability of drawing a blue ball from each box
P_Blue_A = 4 / 12
P_Blue_B = 7 / 15
P_Blue_C = 2 / 15

#Total probability of drawing a blue ball
P_Blue = (
    P_Blue_A * P_A +
    P_Blue_B * P_B +
    P_Blue_C * P_C
)

# Bayes' Theorem
P_C_given_Blue = (P_Blue_C * P_C) / P_Blue

print("Probability of selecting Box C given a blue ball:")
print(f"{P_C_given_Blue:.6f}")
print(f"Fraction: {2/29}")