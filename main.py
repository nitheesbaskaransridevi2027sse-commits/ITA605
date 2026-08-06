import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from neural_network import NeuralNetwork


# Load Iris Dataset

iris = load_iris()

X = iris.data

y = iris.target.reshape(-1, 1)

# Normalize inputs

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# One Hot Encode outputs

encoder = OneHotEncoder(sparse_output=False)

y = encoder.fit_transform(y)

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Neural Network

nn = NeuralNetwork(
    input_size=4,
    hidden_size=8,
    output_size=3,
    learning_rate=0.5
)

# Train

losses = nn.train(
    X_train,
    y_train,
    epochs=1000
)

# Test

predictions = nn.predict(X_test)

actual = y_test.argmax(axis=1)

accuracy = (predictions == actual).mean() * 100

print()

print("Testing Accuracy : {:.2f}%".format(accuracy))

# Plot Loss

plt.plot(losses)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()
