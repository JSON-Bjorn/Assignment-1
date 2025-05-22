import math

class NeuronManual:
    """Implements a single neuron manually without NumPy."""

    def __init__(self, num_inputs, activation_fn='sigmoid'):
        """Initializes the neuron.

        Args:
            num_inputs (int): The number of inputs the neuron will receive.
            activation_fn (str): The activation function to use ('sigmoid', 'relu', or None).
        """
        # Initialize weights and bias (e.g., with small random values or zeros)
        # For simplicity, let's use zeros here. In practice, random initialization is better.
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.activation_fn = activation_fn

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        # Add a small epsilon to prevent overflow with large negative inputs
        epsilon = 1e-15
        x = max(x, -700) # Avoid math.exp overflow for large negative numbers
        return 1 / (1 + math.exp(-x + epsilon))

    def _relu(self, x):
        """ReLU activation function."""
        return max(0, x)

    def forward(self, inputs):
        """Performs the forward pass of the neuron.

        Args:
            inputs (list or tuple): A list or tuple of input values.

        Returns:
            float: The output of the neuron after applying the activation function.

        Raises:
            ValueError: If the number of inputs doesn't match the neuron's expectation.
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, but got {len(inputs)}")

        # Calculate the weighted sum (dot product) + bias
        weighted_sum = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias

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
    neuron = NeuronManual(num_inputs=3, activation_fn='sigmoid')

    # Set some example weights and bias
    neuron.weights = [0.5, -1.0, 0.1]
    neuron.bias = 0.2

    # Example input
    inputs = [1.0, 2.0, 3.0]

    # Calculate the output
    output = neuron.forward(inputs)
    print(f"Inputs: {inputs}")
    print(f"Weights: {neuron.weights}")
    print(f"Bias: {neuron.bias}")
    print(f"Activation: {neuron.activation_fn}")
    print(f"Output: {output:.4f}") # Expected: sigmoid(1*0.5 + 2*(-1) + 3*0.1 + 0.2) = sigmoid(-1.0) approx 0.2689

    # Example with ReLU
    neuron_relu = NeuronManual(num_inputs=2, activation_fn='relu')
    neuron_relu.weights = [-0.2, 0.8]
    neuron_relu.bias = -0.1
    inputs_relu = [4.0, 1.0]
    output_relu = neuron_relu.forward(inputs_relu)
    print("\n--- ReLU Example ---")
    print(f"Inputs: {inputs_relu}")
    print(f"Weights: {neuron_relu.weights}")
    print(f"Bias: {neuron_relu.bias}")
    print(f"Activation: {neuron_relu.activation_fn}")
    print(f"Output: {output_relu:.4f}") # Expected: relu(4*(-0.2) + 1*0.8 - 0.1) = relu(-0.1) = 0.0