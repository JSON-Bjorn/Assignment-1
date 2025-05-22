import numpy as np

class NeuronNumPy:
    """Implements a single neuron using NumPy for efficient vector operations."""

    def __init__(self, num_inputs, activation_fn='sigmoid'):
        """Initializes the neuron.

        Args:
            num_inputs (int): The number of inputs the neuron will receive.
            activation_fn (str): The activation function to use ('sigmoid', 'relu', or None).
        """
        # Initialize weights and bias with small random values for better learning
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = np.zeros(1)
        self.activation_fn = activation_fn

    def _sigmoid(self, x):
        """Sigmoid activation function using NumPy."""
        # Apply clipping to prevent overflow/underflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        """ReLU activation function using NumPy."""
        return np.maximum(0, x)

    def forward(self, inputs):
        """Performs the forward pass of the neuron using NumPy.

        Args:
            inputs (np.ndarray): A NumPy array of input values (shape: (num_inputs,)).

        Returns:
            np.ndarray: The output of the neuron (a single value in a NumPy array).

        Raises:
            ValueError: If the input shape doesn't match the neuron's weights.
        """
        if inputs.shape[0] != self.weights.shape[0]:
            raise ValueError(f"Expected input shape ({self.weights.shape[0]},), but got {inputs.shape}")

        # Calculate the weighted sum (dot product) + bias using NumPy
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply activation function
        if self.activation_fn == 'sigmoid':
            output = self._sigmoid(weighted_sum)
        elif self.activation_fn == 'relu':
            output = self._relu(weighted_sum)
        elif self.activation_fn is None:
            output = weighted_sum # Linear activation
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        return output

# Example Usage (Optional)
if __name__ == '__main__':
    # Example: Neuron with 3 inputs and sigmoid activation
    neuron = NeuronNumPy(num_inputs=3, activation_fn='sigmoid')

    # Example input (as a NumPy array)
    inputs = np.array([1.0, 2.0, 3.0])

    # Calculate the output
    output = neuron.forward(inputs)
    print(f"Inputs: {inputs}")
    print(f"Weights: {neuron.weights}")
    print(f"Bias: {neuron.bias}")
    print(f"Activation: {neuron.activation_fn}")
    print(f"Output: {output[0]:.4f}") # Output depends on random weights

    # Example with ReLU
    neuron_relu = NeuronNumPy(num_inputs=2, activation_fn='relu')
    inputs_relu = np.array([4.0, 1.0])
    output_relu = neuron_relu.forward(inputs_relu)
    print("\n--- ReLU Example ---")
    print(f"Inputs: {inputs_relu}")
    print(f"Weights: {neuron_relu.weights}")
    print(f"Bias: {neuron_relu.bias}")
    print(f"Activation: {neuron_relu.activation_fn}")
    print(f"Output: {output_relu[0]:.4f}") # Output depends on random weights