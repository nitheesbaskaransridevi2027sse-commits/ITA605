training_accuracy=[60,65,70,75,80,85,88,92]
validation_accuracy=[58,63,68,72,73,72,70,68]
print("Epoch\tTraining Accuracy\tValidation Accuracy")
for i in range(len(training_accuracy)):
    print(f"{i+1}\t{training_accuracy[i]}%\t\t\t{validation_accuracy[i]}%")
overfit_epoch=0
for i in range(1,len(validation_accuracy)):
    if validation_accuracy[i]<validation_accuracy[i-1] and training_accuracy[i]>training_accuracy[i-1]:
        overfit_epoch=i+1
        break
print("\nOverfitting begins at Epoch:",overfit_epoch)
print("\nReason:")
print("Validation accuracy decreases because the model memorizes the training data and fails to generalize to unseen data.")
print("\nMethods to improve validation performance:")
print("1. Early Stopping")
print("2. Dropout or Data Augmentation")