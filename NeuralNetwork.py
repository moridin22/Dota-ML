import numpy as np
import pandas as pd

#
# Helper functions
#

def add_bias(inputs):
    """Returns the list of inputs with a bias column."""
    return np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

def biased(inputs):
    """A necessary but not sufficient condition for the inputs to be biased."""
    return (inputs[:,-1] == np.ones((inputs.shape[0], 1))).all()

#
# The main class
#

class NeuralNetwork:
    """A neural network with one hidden layer using the backpropagation algorithm."""

    def __init__(self, shape, momentum = 0, learning_rate = 1):
        """Initializes the network.

        The SHAPE parameter should be a tuple with the number of input columns
        (including the bias column!!) as the first element, 1 as the last element,
        and the number of hidden neurons in each layer as the middle elements.
        """

        self.shape = shape
        self.momentum = momentum
        self.learning_rate = learning_rate

        # Initialize the weights to random values
        self.first_weights = 2 * np.random.random((self.shape[0], self.shape[1])) - 1
        self.second_weights = 2 * np.random.random((self.shape[1], self.shape[2])) - 1

    def sigmoid(self, x, deriv = False):
        """Returns the sigmoid function or its derivative of the input list."""
        if deriv:
            return x * (1-x)
        else:
            return 1 / (1 + np.exp(-x))

    def run(self, inputs):
        """Runs the neural network on a set of inputs."""
        assert biased(inputs), "Inputs need a bias column"
        self.first_layer_output = self.sigmoid(np.dot(inputs, self.first_weights))
        self.second_layer_output = self.sigmoid(np.dot(self.first_layer_output, self.second_weights))

    def print_error(self):
        print("Error:" + str(np.mean(np.abs(self.error))))

    def train(self, inputs, target, num_loops = 5000, num_error_prints = 10):
        """Train the network on the INPUTS dataset (should be a numpy array)."""

        assert biased(inputs), "Inputs need a bias column"
        assert
        first_past_update = second_past_update = 0
        error_mod = num_loops // num_error_prints
        for j in range(num_loops):
            self.run(inputs)
            self.error = target - self.second_layer_output

            # Print the error every ERROR_MOD iterations
            if (j % error_mod) == 0:
                self.print_error()

            # Determine the update amounts
            second_delta = (self.error) * (self.sigmoid(self.second_layer_output, deriv = True))
            first_layer_error = (second_delta).dot(self.second_weights.T)
            first_delta = first_layer_error * (self.sigmoid(self.first_layer_output, deriv = True))
            first_update = self.learning_rate * inputs.T.dot(first_delta) + self.momentum * first_past_update
            second_update = self.learning_rate * self.first_layer_output.T.dot(second_delta) + self.momentum * second_past_update

            # Update the weights
            self.first_weights += first_update
            self.second_weights += second_update
            first_past_update, second_past_update = first_update.copy(), second_update.copy()

def test():
    X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    y = np.array([[0,1,1,0]]).T
    X = add_bias(X)
    nn = NeuralNetwork((4,4,1))
    nn.train(X,y)
