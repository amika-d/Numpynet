import numpy as np

class XORNet:
    """
    A 2-Layer Neural Network built from scratch using NumPy.
    Solves the XOR problem to demonstrate Backpropagation.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.uniform(size=(input_size, hidden_size))
        self.b1 = np.random.uniform(size=(1, hidden_size))
        self.W2 = np.random.uniform(size=(hidden_size, output_size))
        self.b2 = np.random.uniform(size=(1, output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # Assumes x is already the sigmoid output
        return x * (1 - x)

    def forward(self, X):
        # Layer 1
        self.h1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.h1)
        
        # Layer 2
        self.h2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self._sigmoid(self.h2)
        
        return self.y_hat

    def backward(self, X, y, lr=0.1):
        # 1. Output Delta
        delta_out = (self.y_hat - y) * self._sigmoid_derivative(self.y_hat)
        
        # 2. Hidden Delta
        error_hidden = delta_out.dot(self.W2.T)
        delta_hidden = error_hidden * self._sigmoid_derivative(self.a1)
        
        # 3. Gradients
        grad_W2 = self.a1.T.dot(delta_out)
        grad_W1 = X.T.dot(delta_hidden)
        
        # 4. Updates
        self.W2 -= lr * grad_W2
        self.W1 -= lr * grad_W1
        self.b2 -= lr * np.sum(delta_out, axis=0, keepdims=True)
        self.b1 -= lr * np.sum(delta_hidden, axis=0, keepdims=True)