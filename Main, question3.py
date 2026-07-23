TP = 210
TN = 190
FP = 30
samples_per_class = 1800
test_ratio = 0.20
positive_test = samples_per_class * test_ratio
negative_test = samples_per_class * test_ratio
FN = int(positive_test - TP)
total_test = positive_test + negative_test
accuracy = (TP + TN) / total_test
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Confusion Matrix")
print("----------------")
print(f"True Positive (TP): {TP}")
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print("\nPerformance Metrics")
print("-------------------")
print(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision   : {precision:.4f} ({precision*100:.2f}%)")
print(f"Sensitivity : {recall:.4f} ({recall*100:.2f}%)")
print(f"Specificity : {specificity:.4f} ({specificity*100:.2f}%)")