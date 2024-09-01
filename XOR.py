# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining the sigmoid function and its derivative for activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset - representing the four possible combinations of XOR inputs
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# Expected output for XOR
expected_output = np.array([[0], [1], [1], [0]])

# Initializing weights and biases with random values
np.random.seed(1) #rastgele sayıların her sıfırdan başladığında aynı değerleri vermesi sağlanıyor. böylece tekrar kontrol edilebilir ve izlenebilir çözümler oluşuyor
hidden_weights = np.random.uniform(size=(2, 4))  # 2 inputs, 4 neurons in hidden layer
hidden_bias = np.random.uniform(size=(1, 4))
output_weights = np.random.uniform(size=(4, 1))  # 4 neurons in hidden layer, 1 output
output_bias = np.random.uniform(size=(1, 1))  # Corrected dimension for output bias

# Learning rate
lr = 0.1

# List to store errors
errors = []

# Training the neural network
for epoch in range(10000):
    # Forward propagation
    # Hidden layer activations
    hidden_layer_activation = sigmoid(np.dot(inputs, hidden_weights) + hidden_bias)
    # Output layer activations
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights) + output_bias)

    # Backpropagation
    # Calculating error at the output
    output_error = expected_output - output_layer_activation
    # Applying derivative of sigmoid to error
    output_adjustments = output_error * sigmoid_derivative(output_layer_activation)

    # Calculating error at the hidden layer
    hidden_error = output_adjustments.dot(output_weights.T)
    # Applying derivative of sigmoid to error
    hidden_adjustments = hidden_error * sigmoid_derivative(hidden_layer_activation)

    # Updating weights and biases
    output_weights += hidden_layer_activation.T.dot(output_adjustments) * lr
    output_bias += np.sum(output_adjustments, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(hidden_adjustments) * lr
    hidden_bias += np.sum(hidden_adjustments, axis=0, keepdims=True) * lr

    # Recording error for each epoch
    errors.append(np.mean(np.abs(output_error)))

# Plotting the error over epochs
# Plotting the line graph
plt.plot(errors, label='Error Rate')

# Highlighting the last value
plt.scatter(len(errors)-1, errors[-1], color='red')  # Adding a red marker at the last point
# Formatting the text label to show only three decimal places
plt.text(len(errors)-1, errors[-1], f' {errors[-1]:.3f}', verticalalignment='bottom')

# Adding titles and labels to the graph
plt.title('Error Rate Over Time')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()

# Displaying the graph
plt.show()
