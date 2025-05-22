import numpy as np

class AnnNumPyLayer:
    """Implements a single layer of an Artificial Neural Network using NumPy."""

    def __init__(self, num_inputs, num_neurons, activation_fn='sigmoid'):
        """Initializes the ANN layer.

        Args:
            num_inputs (int): The number of input features for each sample.
            num_neurons (int): The number of neurons in this layer.
            activation_fn (str): The activation function ('sigmoid', 'relu', or None).
        """
        # Initialize weights and biases
        # Weights shape: (num_inputs, num_neurons)
        # Biases shape: (1, num_neurons) - allows broadcasting
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.zeros((1, num_neurons))
        self.activation_fn = activation_fn

    def _sigmoid(self, x):
        """Sigmoid activation function using NumPy."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        """ReLU activation function using NumPy."""
        return np.maximum(0, x)

    def forward(self, inputs):
        """Performs the forward pass for a batch of inputs.

        Args:
            inputs (np.ndarray): A NumPy array of input values.
                                   Shape: (batch_size, num_inputs).

        Returns:
            np.ndarray: The output of the layer after applying activation.
                          Shape: (batch_size, num_neurons).
        """
        # Calculate the linear combination (matrix multiplication + bias)
        # inputs shape: (batch_size, num_inputs)
        # weights shape: (num_inputs, num_neurons)
        # result shape: (batch_size, num_neurons)
        linear_combination = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        if self.activation_fn == 'sigmoid':
            output = self._sigmoid(linear_combination)
        elif self.activation_fn == 'relu':
            output = self._relu(linear_combination)
        elif self.activation_fn is None:
            output = linear_combination # Linear activation
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        return output

# Example Usage (Optional)
if __name__ == '__main__':
    # Example: A layer with 4 inputs and 3 neurons, using ReLU
    num_inputs = 4
    num_neurons = 3
    layer = AnnNumPyLayer(num_inputs, num_neurons, activation_fn='relu')

    # Create a batch of 2 samples
    # Shape: (batch_size, num_inputs) = (2, 4)
    inputs = np.array([
        [1.0, 2.0, 3.0, 4.0],  # Sample 1
        [-1.0, 0.5, -2.0, 1.5]  # Sample 2
    ])

    # Perform the forward pass
    output = layer.forward(inputs)

    print(f"Inputs (shape {inputs.shape}):\n{inputs}")
    print(f"\nWeights (shape {layer.weights.shape}):\n{layer.weights}")
    print(f"\nBiases (shape {layer.biases.shape}):\n{layer.biases}")
    print(f"\nActivation: {layer.activation_fn}")
    print(f"\nOutput (shape {output.shape}):\n{output}")

    # Example with Sigmoid
    layer_sigmoid = AnnNumPyLayer(2, 5, activation_fn='sigmoid') # 2 inputs, 5 neurons
    inputs_sigmoid = np.random.rand(3, 2) # Batch of 3 samples, 2 features each
    output_sigmoid = layer_sigmoid.forward(inputs_sigmoid)
    print("\n--- Sigmoid Example ---")
    print(f"Inputs (shape {inputs_sigmoid.shape}):\n{inputs_sigmoid}")
    print(f"Output (shape {output_sigmoid.shape}):\n{output_sigmoid}")