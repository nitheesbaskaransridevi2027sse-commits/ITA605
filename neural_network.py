import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):

        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)

        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        self.bias_output = np.zeros((1, output_size))

    # Sigmoid activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward propagation
    def forward(self, X):

        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output,
                                  self.weights_hidden_output) + self.bias_output

        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    # Backpropagation
    def backward(self, X, y):

        output_error = y - self.final_output

        output_delta = output_error * self.sigmoid_derivative(self.final_output)

        hidden_error = np.dot(output_delta,
                              self.weights_hidden_output.T)

        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights

        self.weights_hidden_output += self.learning_rate * np.dot(
            self.hidden_output.T,
            output_delta
        )

        self.bias_output += self.learning_rate * np.sum(
            output_delta,
            axis=0,
            keepdims=True
        )

        self.weights_input_hidden += self.learning_rate * np.dot(
            X.T,
            hidden_delta
        )

        self.bias_hidden += self.learning_rate * np.sum(
            hidden_delta,
            axis=0,
            keepdims=True
        )

        loss = np.mean(np.square(output_error))

        return loss

    # Train the network
    def train(self, X, y, epochs):

        losses = []

        for epoch in range(epochs):

            self.forward(X)

            loss = self.backward(X, y)

            losses.append(loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1} Loss : {loss:.4f}")

        return losses

    # Prediction
    def predict(self, X):

        output = self.forward(X)

        return np.argmax(output, axis=1)