import numpy as np

# Define input
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# Define expected output
outputs = np.array([[1],[1],[1],[0]])

# Define weights with bias
weights = np.array([[-2.0],[2.0],[2.0],[1.0]])

# Define sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define learning rate
learning_rate = 0.1

# Train the neural network
for iteration in range(10000):
    # Calculate output of neural network
    weighted_sum = np.dot(inputs, weights)
    activated_output = sigmoid(weighted_sum)

    # Calculate error
    error = activated_output - outputs

    # Adjust weights
    adjustments = error * sigmoid_derivative(activated_output)
    weights -= learning_rate * np.dot(inputs.T, adjustments)

# Predict the output
test_input = np.array([[1,1]])
weighted_sum = np.dot(test_input, weights)
activated_output = sigmoid(weighted_sum)
print("NAND of {} is {}".format(test_input, activated_output))
