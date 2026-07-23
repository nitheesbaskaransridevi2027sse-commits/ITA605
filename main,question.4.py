training_loss=[2.8,2.5,2.2,2.0,1.8,1.6,1.5,1.3]
validation_loss=[2.4,2.2,2.0,2.1,2.3,2.6,2.9,3.2]
print("Epoch\tTraining Loss\tValidation Loss")
for i in range(len(training_loss)):
    print(f"{i+1}\t{training_loss[i]}\t\t{validation_loss[i]}")
overfit_epoch=0
for i in range(1,len(validation_loss)):
    if validation_loss[i]>validation_loss[i-1] and training_loss[i]<training_loss[i-1]:
        overfit_epoch=i+1
        break
print("\nOverfitting begins at Epoch:",overfit_epoch)
print("\nTechniques to reduce overfitting:")
print("1. Early Stopping")
print("2. Dropout or L2 Regularization")
print("\nEffect of Overfitting:")
print("Overfitting reduces the model's ability to generalize to unseen data. The training loss keeps decreasing while the validation loss increases, resulting in poorer prediction accuracy on new data.")