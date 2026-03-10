# Week 1: Neural Networks Basics - Complete Tutorial

## The Foundation of Deep Learning

> **Expert Guide**: 15+ Years of Experience Distilled into One Week
>
> By the end of this week, you'll understand neural networks from first principles to implementation, with crystal-clear intuition and working code.

---

## 📚 Table of Contents

1. [Introduction: What Really Is a Neural Network?](#introduction)
2. [Day 1: The Single Neuron - Building Block of Intelligence](#day-1)
3. [Day 2: Forward Propagation - How Neural Networks Think](#day-2)
4. [Day 3: Backpropagation - How Neural Networks Learn](#day-3)
5. [Day 4: Building Your First Network](#day-4)
6. [Day 5: Introduction to PyTorch](#day-5)
7. [Weekend Project: MNIST Digit Recognition](#weekend-project)
8. [Week Review & Key Takeaways](#week-review)

---

<a name="introduction"></a>

## 🧠 Introduction: What Really Is a Neural Network?

### The Big Picture Intuition

Imagine teaching a child to recognize cats:

- You show them **many examples** of cats
- They learn to identify **patterns** (whiskers, ears, fur)
- Eventually, they can recognize **new cats** they've never seen

This is **exactly** what neural networks do, but mathematically!

### The Mathematical Reality

A neural network is simply a **mathematical function** that:

```
f(input) → output
```

But unlike traditional functions (like `y = mx + b`), neural networks:

1. Have **millions of parameters** (not just `m` and `b`)
2. Learn these parameters from **data** (not from equations)
3. Can approximate **any function** (Universal Approximation Theorem)

### Why "Neural"?

The name comes from biological neurons, but that's where the similarity ends. Here's the truth:

**Biological Neuron:**

- Receives signals through dendrites
- Processes in the cell body
- Fires through axon if threshold reached

**Artificial Neuron:**

- Receives numbers (inputs)
- Multiplies by weights and adds bias
- Applies activation function

They're inspired by biology but are pure mathematics!

---

<a name="day-1"></a>

## 📅 Day 1: The Single Neuron - Building Block of Intelligence

### 1.1 The Simplest Neuron: Linear Combination

Let's start with the absolute basics.

#### Mathematical Definition

A single neuron performs this operation:

$$
y = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

Or in vector form:

$$
y = \mathbf{w}^T\mathbf{x} + b
$$

Where:

- $\mathbf{x}$ = input vector $[x_1, x_2, ..., x_n]$
- $\mathbf{w}$ = weight vector $[w_1, w_2, ..., w_n]$
- $b$ = bias (a single number)
- $y$ = output

#### Intuition: What's Really Happening?

Think of a neuron as a **decision maker**:

**Example: Should I carry an umbrella?**

Inputs:

- $x_1$ = Rain probability (0 to 1)
- $x_2$ = Cloud coverage (0 to 1)
- $x_3$ = Wind speed (0 to 1, normalized)

Weights (importance):

- $w_1 = 0.8$ (rain probability is very important!)
- $w_2 = 0.3$ (clouds matter, but less)
- $w_3 = 0.1$ (wind doesn't matter much)

Bias:

- $b = -0.5$ (I'm optimistic, need strong signal to carry umbrella)

**Calculation:**

```python
rain_prob = 0.7
clouds = 0.6
wind = 0.4

y = 0.8*0.7 + 0.3*0.6 + 0.1*0.4 + (-0.5)
y = 0.56 + 0.18 + 0.04 - 0.5
y = 0.28
```

If `y > 0`: Carry umbrella! ☂️

#### Code Implementation (Pure Python)

```python
import numpy as np

class SingleNeuron:
    """
    A single neuron implementation from scratch.

    This is the fundamental building block of all neural networks.
    """

    def __init__(self, n_inputs):
        """
        Initialize neuron with random weights and zero bias.

        Args:
            n_inputs (int): Number of input features
        """
        # Initialize weights randomly (small values)
        self.weights = np.random.randn(n_inputs) * 0.01
        # Initialize bias to zero
        self.bias = 0.0

    def forward(self, x):
        """
        Compute the output of the neuron.

        Args:
            x (np.array): Input vector of shape (n_inputs,)

        Returns:
            float: Output of the neuron
        """
        # Weighted sum + bias
        return np.dot(self.weights, x) + self.bias

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"


# Example usage
if __name__ == "__main__":
    # Create a neuron with 3 inputs
    neuron = SingleNeuron(n_inputs=3)

    print("Initial neuron state:")
    print(neuron)
    print()

    # Umbrella example
    weather_data = np.array([0.7, 0.6, 0.4])  # [rain, clouds, wind]
    output = neuron.forward(weather_data)

    print(f"Input: {weather_data}")
    print(f"Output: {output:.4f}")
    print(f"Decision: {'Carry umbrella' if output > 0 else 'No umbrella needed'}")
```

### 1.2 Activation Functions: Making It Nonlinear

#### The Problem with Linear Neurons

If we only use linear operations, our network can only learn **linear relationships**:

```python
# Two linear neurons
y1 = w1*x + b1
y2 = w2*y1 + b2
# Simplifies to:
y2 = (w2*w1)*x + (w2*b1 + b2)
# Still just: y = mx + b
```

**No matter how many layers, it's still just a line!**

Real-world problems are **nonlinear**:

- Is this a cat or dog? (No linear boundary)
- What's the next word? (Language is nonlinear)
- Will stock price go up? (Markets are nonlinear)

#### Solution: Activation Functions

We add a **nonlinear function** after the weighted sum:

$$
y = \sigma(w^Tx + b)
$$

Where $\sigma$ is the activation function.

#### Common Activation Functions

**1. Sigmoid Function**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Properties:**

- Output range: (0, 1)
- S-shaped curve
- Smooth and differentiable

**When to use:**

- Binary classification (output layer)
- When you need probabilities

**Intuition:** Squashes any number into a probability between 0 and 1.

```python
def sigmoid(z):
    """
    Sigmoid activation function.

    Squashes input to range (0, 1).
    Useful for binary classification.

    Args:
        z: Input value or array

    Returns:
        Activated value between 0 and 1
    """
    return 1 / (1 + np.exp(-z))

# Visualization
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.grid(True)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Activation Function')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary')
plt.legend()
plt.show()
```

**2. Tanh (Hyperbolic Tangent)**

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

**Properties:**

- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Steeper gradient than sigmoid

**When to use:**

- Hidden layers (better than sigmoid)
- When outputs should be negative/positive

```python
def tanh(z):
    """
    Tanh activation function.

    Similar to sigmoid but outputs range from -1 to 1.
    Zero-centered, which helps with training.

    Args:
        z: Input value or array

    Returns:
        Activated value between -1 and 1
    """
    return np.tanh(z)

# Comparison
z = np.linspace(-5, 5, 100)
plt.plot(z, sigmoid(z), label='Sigmoid')
plt.plot(z, tanh(z), label='Tanh')
plt.grid(True)
plt.xlabel('z')
plt.ylabel('Activation')
plt.title('Sigmoid vs Tanh')
plt.legend()
plt.show()
```

**3. ReLU (Rectified Linear Unit)** ⭐ Most Popular!

$$
\text{ReLU}(z) = \max(0, z)
$$

**Properties:**

- Output range: [0, ∞)
- Dead simple: if negative, output 0; else output z
- No vanishing gradient problem
- Sparse activation (many neurons output 0)

**When to use:**

- Default choice for hidden layers
- Almost always the best starting point

**Intuition:** "Only positive information passes through"

```python
def relu(z):
    """
    ReLU activation function.

    The most popular activation function.
    Simple: output the input if positive, else 0.

    Args:
        z: Input value or array

    Returns:
        max(0, z)
    """
    return np.maximum(0, z)

# Why ReLU is powerful
z = np.linspace(-5, 5, 100)
plt.plot(z, relu(z), linewidth=2)
plt.grid(True)
plt.xlabel('z')
plt.ylabel('ReLU(z)')
plt.title('ReLU: Simple Yet Powerful')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.show()
```

**4. Leaky ReLU** (ReLU with small negative slope)

$$
\text{LeakyReLU}(z) = \max(0.01z, z)
$$

**Why?** Fixes the "dying ReLU" problem (neurons that never activate).

```python
def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU activation function.

    Allows small negative values, preventing "dead" neurons.

    Args:
        z: Input value or array
        alpha: Slope for negative values (default 0.01)

    Returns:
        z if z > 0, else alpha * z
    """
    return np.where(z > 0, z, alpha * z)
```

**5. Softmax** (For multi-class classification)

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties:**

- Outputs sum to 1 (probability distribution)
- Used in output layer for classification

**Intuition:** "Convert scores to probabilities"

```python
def softmax(z):
    """
    Softmax activation function.

    Converts a vector of numbers into a probability distribution.

    Args:
        z: Input array of shape (n_classes,)

    Returns:
        Array of probabilities that sum to 1
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Example: Classification scores
scores = np.array([2.0, 1.0, 0.1])  # [cat, dog, bird]
probabilities = softmax(scores)

print("Scores:", scores)
print("Probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))
# Output: [0.659, 0.242, 0.099]  -> Most likely a cat!
```

### 1.3 Complete Neuron with Activation

```python
class Neuron:
    """
    A complete artificial neuron with activation function.

    This is what actually gets used in neural networks.
    """

    def __init__(self, n_inputs, activation='relu'):
        """
        Initialize neuron.

        Args:
            n_inputs (int): Number of input features
            activation (str): Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.activation = activation

    def activate(self, z):
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z  # Linear

    def forward(self, x):
        """
        Forward pass through the neuron.

        Args:
            x (np.array): Input vector

        Returns:
            float: Activated output
        """
        # Linear combination
        z = np.dot(self.weights, x) + self.bias
        # Nonlinear activation
        a = self.activate(z)
        return a


# Demonstration
neuron_relu = Neuron(n_inputs=3, activation='relu')
neuron_sigmoid = Neuron(n_inputs=3, activation='sigmoid')

x = np.array([1.0, -0.5, 2.0])

print(f"Input: {x}")
print(f"ReLU output: {neuron_relu.forward(x):.4f}")
print(f"Sigmoid output: {neuron_sigmoid.forward(x):.4f}")
```

### 1.4 Visual Understanding

Let's visualize how a single neuron creates a **decision boundary**:

```python
import matplotlib.pyplot as plt

def visualize_neuron_decision():
    """
    Visualize how a single neuron separates data.

    This shows the fundamental concept of classification.
    """
    # Create a simple dataset (2D for visualization)
    np.random.seed(42)

    # Class 0: Points in bottom-left
    class_0 = np.random.randn(50, 2) + np.array([-2, -2])
    # Class 1: Points in top-right
    class_1 = np.random.randn(50, 2) + np.array([2, 2])

    # Single neuron
    neuron = Neuron(n_inputs=2, activation='sigmoid')
    # Set weights manually for clear separation
    neuron.weights = np.array([1.0, 1.0])
    neuron.bias = 0.0

    # Create grid for decision boundary
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Calculate output for each point in grid
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            Z[i, j] = neuron.forward(point)

    # Plot
    plt.figure(figsize=(10, 8))

    # Decision boundary (where output = 0.5)
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='Neuron Output')

    # Plot data points
    plt.scatter(class_0[:, 0], class_0[:, 1], c='blue',
                marker='o', s=50, label='Class 0', edgecolors='k')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='red',
                marker='s', s=50, label='Class 1', edgecolors='k')

    # Decision boundary line (output = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)

    plt.xlabel('Feature 1 (x₁)')
    plt.ylabel('Feature 2 (x₂)')
    plt.title('Single Neuron Decision Boundary\n' +
              f'w₁={neuron.weights[0]}, w₂={neuron.weights[1]}, b={neuron.bias}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# Run visualization
visualize_neuron_decision()
```

### 1.5 Key Insights from Day 1

✅ **A neuron is just weighted sum + activation**

- Linear part: $z = w^Tx + b$
- Nonlinear part: $a = \sigma(z)$

✅ **Weights control importance of inputs**

- Larger weight = input matters more
- Negative weight = input has opposite effect

✅ **Bias allows shifting the decision boundary**

- Positive bias = easier to activate
- Negative bias = harder to activate

✅ **Activation functions introduce nonlinearity**

- Without them, network is just a linear function
- ReLU is default choice (simple and effective)

✅ **A single neuron creates a linear decision boundary**

- In 2D: a line
- In 3D: a plane
- In n-D: a hyperplane

---

<a name="day-2"></a>

## 📅 Day 2: Forward Propagation - How Neural Networks Think

### 2.1 From Single Neuron to Layer

One neuron is limited - it can only create **one linear boundary**.

**Question:** How do we solve complex problems?
**Answer:** Use **multiple neurons** organized in **layers**!

#### Layer Architecture

A **layer** is a collection of neurons that:

1. All receive the **same inputs**
2. Each has **different weights** (learns different patterns)
3. Produce **multiple outputs** (one per neuron)

```
Input: x = [x₁, x₂, x₃]
         ↓  ↓  ↓
      [Neuron 1] → a₁
      [Neuron 2] → a₂
      [Neuron 3] → a₃
      [Neuron 4] → a₄

Output: a = [a₁, a₂, a₃, a₄]
```

#### Mathematical Formulation

For a layer with $n$ inputs and $m$ neurons:

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

$$
\mathbf{a} = \sigma(\mathbf{z})
$$

Where:

- $\mathbf{x}$ ∈ ℝⁿ (input vector)
- $\mathbf{W}$ ∈ ℝᵐˣⁿ (weight matrix)
- $\mathbf{b}$ ∈ ℝᵐ (bias vector)
- $\mathbf{z}$ ∈ ℝᵐ (pre-activation)
- $\mathbf{a}$ ∈ ℝᵐ (activation output)

#### Matrix Multiplication Intuition

Let's break down what $\mathbf{W}\mathbf{x}$ really means:

```python
# Example: 3 inputs, 4 neurons
W = np.array([
    [w11, w12, w13],  # Weights for neuron 1
    [w21, w22, w23],  # Weights for neuron 2
    [w31, w32, w33],  # Weights for neuron 3
    [w41, w42, w43]   # Weights for neuron 4
])

x = np.array([x1, x2, x3])

# Matrix multiplication computes all neurons at once!
z = W @ x  # @ is matrix multiplication
# z[0] = w11*x1 + w12*x2 + w13*x3  (neuron 1)
# z[1] = w21*x1 + w22*x2 + w23*x3  (neuron 2)
# z[2] = w31*x1 + w32*x2 + w33*x3  (neuron 3)
# z[3] = w41*x1 + w42*x2 + w43*x3  (neuron 4)
```

**Key Insight:** Matrix multiplication is **parallel** neuron computation!

### 2.2 Building a Layer Class

```python
class Layer:
    """
    A complete neural network layer.

    This is the fundamental building block of deep networks.
    """

    def __init__(self, n_inputs, n_neurons, activation='relu'):
        """
        Initialize layer with random weights.

        Args:
            n_inputs (int): Number of input features
            n_neurons (int): Number of neurons in this layer
            activation (str): Activation function
        """
        # He initialization for ReLU, Xavier for tanh/sigmoid
        if activation == 'relu':
            # He initialization: std = sqrt(2/n_inputs)
            self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        else:
            # Xavier initialization: std = sqrt(1/n_inputs)
            self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(1.0 / n_inputs)

        self.biases = np.zeros((n_neurons, 1))
        self.activation = activation

        # For backpropagation (later)
        self.z = None
        self.a = None

    def activate(self, z):
        """Apply activation function element-wise."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            return z  # Linear

    def forward(self, x):
        """
        Forward pass through layer.

        Args:
            x (np.array): Input of shape (n_inputs, batch_size)

        Returns:
            np.array: Output of shape (n_neurons, batch_size)
        """
        # Store for backprop
        self.x = x

        # Linear transformation
        self.z = self.weights @ x + self.biases

        # Nonlinear activation
        self.a = self.activate(self.z)

        return self.a

    def __repr__(self):
        return (f"Layer(inputs={self.weights.shape[1]}, "
                f"neurons={self.weights.shape[0]}, "
                f"activation='{self.activation}')")


# Example: Create a layer
layer = Layer(n_inputs=3, n_neurons=4, activation='relu')
print(layer)
print(f"\nWeight matrix shape: {layer.weights.shape}")
print(f"Bias vector shape: {layer.biases.shape}")

# Forward pass
x = np.array([[1.0], [2.0], [3.0]])  # Single sample
output = layer.forward(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")
```

### 2.3 Multi-Layer Networks (Deep Learning!)

Now we connect multiple layers:

```
Input → Layer 1 → Layer 2 → ... → Layer L → Output
```

Each layer:

- Takes previous layer's output as input
- Applies transformation
- Passes to next layer

This is **forward propagation**!

#### Why "Deep"?

- **Shallow network:** 1-2 hidden layers
- **Deep network:** 3+ hidden layers

**Deep networks learn hierarchical features:**

- Early layers: Simple features (edges, colors)
- Middle layers: Complex features (textures, parts)
- Late layers: High-level concepts (objects, faces)

### 2.4 Complete Neural Network Class

```python
class NeuralNetwork:
    """
    A complete feedforward neural network.

    This is a production-ready implementation from scratch!
    """

    def __init__(self, layer_sizes, activations):
        """
        Initialize network architecture.

        Args:
            layer_sizes (list): Number of neurons in each layer
                               [n_inputs, hidden1, hidden2, ..., n_outputs]
            activations (list): Activation for each layer after input
                               Length should be len(layer_sizes) - 1

        Example:
            # 3 inputs -> 4 hidden -> 2 outputs
            nn = NeuralNetwork([3, 4, 2], ['relu', 'softmax'])
        """
        self.layers = []

        # Create layers
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)

        print(f"Created network with {len(self.layers)} layers:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}: {layer}")

    def forward(self, x):
        """
        Forward propagation through entire network.

        Args:
            x (np.array): Input of shape (n_inputs, batch_size)

        Returns:
            np.array: Final output
        """
        # Pass through each layer sequentially
        a = x
        for layer in self.layers:
            a = layer.forward(a)

        return a

    def predict(self, x):
        """
        Make predictions (same as forward, but clearer name).

        Args:
            x (np.array): Input features

        Returns:
            np.array: Predictions
        """
        return self.forward(x)


# Example: Create a 3-layer network
network = NeuralNetwork(
    layer_sizes=[3, 5, 4, 2],  # 3 inputs, 2 hidden layers, 2 outputs
    activations=['relu', 'relu', 'softmax']
)

# Forward pass with single sample
x = np.array([[1.0], [2.0], [3.0]])
output = network.forward(x)

print(f"\nInput:\n{x}")
print(f"\nOutput (probabilities):\n{output}")
print(f"Sum of probabilities: {np.sum(output):.6f}")
```

### 2.5 Batch Processing

In practice, we process **multiple samples** at once (batching):

```python
def demo_batch_processing():
    """
    Demonstrate efficient batch processing.

    This is how real networks process data!
    """
    network = NeuralNetwork(
        layer_sizes=[2, 3, 1],
        activations=['relu', 'sigmoid']
    )

    # Create batch of 5 samples
    batch_size = 5
    X = np.random.randn(2, batch_size)  # Shape: (n_features, n_samples)

    print(f"Input batch shape: {X.shape}")
    print(f"Input batch:\n{X}\n")

    # Forward pass (processes all samples in parallel!)
    predictions = network.forward(X)

    print(f"Output batch shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")

    # Each column is one sample's prediction
    for i in range(batch_size):
        print(f"Sample {i+1}: {predictions[0, i]:.4f}")

demo_batch_processing()
```

**Why batching?**

1. **Efficiency:** Matrix operations are parallelized (GPU acceleration)
2. **Stability:** Gradient estimates are more stable
3. **Memory:** Can process more data than fits in memory

### 2.6 Visualizing Forward Propagation

Let's see what happens inside the network:

```python
def visualize_forward_pass():
    """
    Visualize activations at each layer.

    This helps understand what the network is "thinking".
    """
    # Create simple network
    network = NeuralNetwork(
        layer_sizes=[2, 4, 3, 1],
        activations=['relu', 'relu', 'sigmoid']
    )

    # Single input
    x = np.array([[0.5], [0.8]])

    # Forward pass and collect activations
    activations = [x]
    a = x
    for layer in network.layers:
        a = layer.forward(a)
        activations.append(a)

    # Plot
    fig, axes = plt.subplots(1, len(activations), figsize=(15, 4))

    for i, (ax, activation) in enumerate(zip(axes, activations)):
        if i == 0:
            title = 'Input'
        elif i == len(activations) - 1:
            title = 'Output'
        else:
            title = f'Layer {i}'

        # Visualize as heatmap
        im = ax.imshow(activation, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Neuron')

        # Add values
        for j in range(activation.shape[0]):
            ax.text(0, j, f'{activation[j, 0]:.2f}',
                   ha='center', va='center')

    plt.colorbar(im, ax=axes, label='Activation')
    plt.tight_layout()
    plt.savefig('forward_propagation_visualization.png', dpi=150)
    plt.show()

visualize_forward_pass()
```

### 2.7 Forward Pass Algorithm (Step-by-Step)

Here's the complete forward propagation algorithm:

```python
def forward_propagation_detailed(network, X):
    """
    Detailed forward propagation with full explanation.

    Args:
        network: NeuralNetwork instance
        X: Input data (n_features, n_samples)

    Returns:
        Final output and all intermediate values
    """
    print("="*60)
    print("FORWARD PROPAGATION - DETAILED WALKTHROUGH")
    print("="*60)

    # Initialize
    A = X
    cache = {'A0': A}  # Store all activations

    print(f"\nInput (A⁰):")
    print(f"  Shape: {A.shape}")
    print(f"  Values:\n{A}\n")

    # Pass through each layer
    for l, layer in enumerate(network.layers, 1):
        print(f"Layer {l} ({layer.activation}):")
        print(f"  Input shape: {A.shape}")

        # Linear transformation
        Z = layer.weights @ A + layer.biases
        print(f"  Z{l} = W{l} @ A{l-1} + b{l}")
        print(f"  Z{l} shape: {Z.shape}")
        print(f"  Z{l} sample values: {Z[:, 0]}")

        # Activation
        A = layer.activate(Z)
        print(f"  A{l} = {layer.activation}(Z{l})")
        print(f"  A{l} shape: {A.shape}")
        print(f"  A{l} sample values: {A[:, 0]}\n")

        # Store
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    print(f"Final output (A{len(network.layers)}):")
    print(A)
    print("="*60)

    return A, cache


# Example
network = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])
X = np.array([[0.5], [0.8]])
output, cache = forward_propagation_detailed(network, X)
```

### 2.8 Key Takeaways from Day 2

✅ **Layers transform inputs through matrix operations**

- $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ (linear)
- $\mathbf{a} = \sigma(\mathbf{z})$ (nonlinear)

✅ **Forward propagation is sequential layer application**

- Output of layer $l$ becomes input to layer $l+1$
- Information flows forward only (no loops)

✅ **Batching enables parallel processing**

- Process multiple samples simultaneously
- Each column in matrix is one sample

✅ **Deep networks learn hierarchical representations**

- Early layers: Low-level features
- Later layers: High-level features
- Final layer: Task-specific output

✅ **Matrix multiplication is key to efficiency**

- One operation computes all neurons in parallel
- GPU acceleration makes this extremely fast

---

<a name="day-3"></a>

## 📅 Day 3: Backpropagation - How Neural Networks Learn

> "Backpropagation is just the chain rule from calculus, applied systematically." - Every ML professor ever

### 3.1 The Learning Problem

We have a network that can make predictions. But they're **random** (weights initialized randomly)!

**Goal:** Adjust weights so predictions match true labels.

**Question:** How do we know which direction to adjust weights?
**Answer:** **Backpropagation** - the algorithm that makes deep learning possible!

### 3.2 The Big Picture

Here's the complete learning cycle:

```
1. Forward Pass: Make prediction
   X → Network → Ŷ

2. Compute Loss: How wrong are we?
   L = Loss(Ŷ, Y)

3. Backward Pass: Compute gradients
   ∂L/∂W, ∂L/∂b for all layers

4. Update Weights: Move in direction that reduces loss
   W = W - α * ∂L/∂W
   b = b - α * ∂L/∂b

5. Repeat until loss is small
```

### 3.3 Loss Functions - Measuring Error

Before we can learn, we need to **measure mistakes**.

#### 3.3.1 Mean Squared Error (MSE) - For Regression

**Use case:** Predicting continuous values (house prices, temperature, etc.)

$$
L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**Intuition:** Average of squared differences

```python
def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss.

    Args:
        y_true: True values (n_samples,)
        y_pred: Predicted values (n_samples,)

    Returns:
        float: MSE loss
    """
    return np.mean((y_true - y_pred) ** 2)

# Example
y_true = np.array([3.0, 5.0, 2.0])
y_pred = np.array([2.8, 5.2, 2.1])

loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")
# Small differences → small loss ✓
```

**Gradient of MSE:**

$$
\frac{\partial L_{MSE}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)
$$

#### 3.3.2 Binary Cross-Entropy - For Binary Classification

**Use case:** Yes/no decisions (spam/not spam, cat/dog)

$$
L_{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

**Intuition:**

- If true label is 1: penalize for predicting < 1
- If true label is 0: penalize for predicting > 0

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
        epsilon: Small constant to avoid log(0)

    Returns:
        float: BCE loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

# Example
y_true = np.array([1, 0, 1, 1])
y_pred_good = np.array([0.9, 0.1, 0.85, 0.95])
y_pred_bad = np.array([0.4, 0.6, 0.3, 0.55])

loss_good = binary_cross_entropy(y_true, y_pred_good)
loss_bad = binary_cross_entropy(y_true, y_pred_bad)

print(f"Good predictions loss: {loss_good:.4f}")
print(f"Bad predictions loss: {loss_bad:.4f}")
# Good predictions → lower loss ✓
```

**Gradient of BCE:**

$$
\frac{\partial L_{BCE}}{\partial \hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}
$$

#### 3.3.3 Categorical Cross-Entropy - For Multi-Class Classification

**Use case:** Multiple categories (digit recognition, image classification)

$$
L_{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})
$$

**Intuition:** Sum of losses for each class

```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Categorical cross-entropy loss.

    Args:
        y_true: True labels, one-hot encoded (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        epsilon: Small constant to avoid log(0)

    Returns:
        float: CCE loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example: 3 classes
y_true = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [1, 0, 0]   # Class 0
])
y_pred = np.array([
    [0.8, 0.1, 0.1],  # Correct!
    [0.2, 0.7, 0.1],  # Correct!
    [0.3, 0.4, 0.3]   # Wrong!
])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"CCE Loss: {loss:.4f}")
```

### 3.4 Gradient Descent - The Optimization Algorithm

Once we have gradients, we update weights:

$$
W^{new} = W^{old} - \alpha \frac{\partial L}{\partial W}
$$

Where $\alpha$ is the **learning rate**.

#### Intuition: Hiking Down a Mountain

Imagine you're on a mountain in fog (can't see far):

- **Goal:** Reach the valley (minimum loss)
- **Strategy:** Feel the slope (gradient) and step downhill
- **Step size:** Learning rate (too big = overshoot, too small = slow)

```python
def gradient_descent_demo():
    """
    Visualize gradient descent on a simple 1D function.
    """
    # Simple function: f(x) = x^2
    def f(x):
        return x ** 2

    def gradient_f(x):
        return 2 * x

    # Start at x = 4
    x = 4.0
    learning_rate = 0.1
    history = [x]

    # Take 20 steps
    for _ in range(20):
        grad = gradient_f(x)
        x = x - learning_rate * grad
        history.append(x)

    # Plot
    x_plot = np.linspace(-5, 5, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f(x_plot), 'b-', linewidth=2, label='f(x) = x²')
    plt.plot(history, [f(x) for x in history], 'ro-',
             markersize=8, label='Gradient descent path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Started at x = {history[0]}")
    print(f"Ended at x = {history[-1]:.6f}")
    print(f"Minimum at x = 0 ✓")

gradient_descent_demo()
```

### 3.5 The Chain Rule - Mathematical Foundation

Backpropagation is just **systematic application of the chain rule**.

#### Chain Rule Refresher

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

**Example:**

$$
y = (2x + 1)^2
$$

Let $g = 2x + 1$, so $y = g^2$

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = 2g \cdot 2 = 4(2x + 1)
$$

#### Chain Rule in Neural Networks

For a 2-layer network:

```
x → z₁ = W₁x + b₁ → a₁ = σ(z₁) → z₂ = W₂a₁ + b₂ → a₂ = σ(z₂) → L
```

To find $\frac{\partial L}{\partial W_1}$, we chain through:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}
$$

This is **backpropagation**: propagating gradients backward!

### 3.6 Backpropagation Algorithm (Detailed)

Let's derive it step by step for a simple network.

#### Network Architecture

```
Input (x) → Hidden Layer → Output Layer → Loss
           W₁, b₁        W₂, b₂
```

#### Forward Pass Equations

$$
z_1 = W_1 x + b_1
$$

$$
a_1 = \sigma(z_1)
$$

$$
z_2 = W_2 a_1 + b_2
$$

$$
a_2 = \sigma(z_2)
$$

$$
L = \frac{1}{2}(a_2 - y)^2 \quad \text{(MSE loss)}
$$

#### Backward Pass Derivation

**Step 1:** Gradient of loss w.r.t. output

$$
\frac{\partial L}{\partial a_2} = a_2 - y
$$

**Step 2:** Gradient w.r.t. $z_2$ (use chain rule)

$$
\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} = (a_2 - y) \cdot \sigma'(z_2)
$$

**Step 3:** Gradient w.r.t. $W_2$ and $b_2$

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot a_1^T
$$

$$
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}
$$

**Step 4:** Gradient w.r.t. $a_1$ (propagate backward)

$$
\frac{\partial L}{\partial a_1} = W_2^T \cdot \frac{\partial L}{\partial z_2}
$$

**Step 5:** Gradient w.r.t. $z_1$

$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \sigma'(z_1)
$$

**Step 6:** Gradient w.r.t. $W_1$ and $b_1$

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T
$$

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}
$$

### 3.7 Activation Function Derivatives

We need derivatives of activation functions:

#### ReLU Derivative

$$
\text{ReLU}(z) = \max(0, z)
$$

$$
\text{ReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

```python
def relu_derivative(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)
```

#### Sigmoid Derivative

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**Beautiful property:** Derivative expressed in terms of function itself!

```python
def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)
```

#### Tanh Derivative

$$
\tanh'(z) = 1 - \tanh^2(z)
$$

```python
def tanh_derivative(z):
    """Derivative of tanh."""
    t = np.tanh(z)
    return 1 - t**2
```

### 3.8 Complete Backpropagation Implementation

```python
class NeuralNetworkWithBackprop:
    """
    Neural network with backpropagation.

    This is a complete, working implementation!
    """

    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        """
        Initialize network.

        Args:
            layer_sizes: [n_inputs, hidden1, ..., n_outputs]
            activations: Activation for each layer
            learning_rate: Step size for gradient descent
        """
        self.layers = []
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

    def forward(self, X):
        """Forward pass - already implemented."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def activation_derivative(self, z, activation):
        """
        Compute derivative of activation function.

        Args:
            z: Pre-activation values
            activation: Activation function name

        Returns:
            Derivative evaluated at z
        """
        if activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:  # linear
            return np.ones_like(z)

    def compute_loss(self, Y_true, Y_pred, loss_type='mse'):
        """
        Compute loss and its derivative.

        Args:
            Y_true: True labels
            Y_pred: Predictions
            loss_type: 'mse' or 'cross_entropy'

        Returns:
            loss (float), dL/dY_pred (array)
        """
        if loss_type == 'mse':
            loss = np.mean((Y_pred - Y_true)**2)
            dL_dY = 2 * (Y_pred - Y_true) / Y_true.shape[1]
        elif loss_type == 'cross_entropy':
            # Assuming softmax output + cross-entropy
            # Combined derivative is simply: Y_pred - Y_true
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-15), axis=0))
            dL_dY = Y_pred - Y_true

        return loss, dL_dY

    def backward(self, Y_true, loss_type='mse'):
        """
        Backward pass - compute all gradients.

        Args:
            Y_true: True labels (same shape as network output)
            loss_type: Type of loss function
        """
        # Get final layer output
        A_final = self.layers[-1].a

        # Compute initial gradient
        loss, dL_dA = self.compute_loss(Y_true, A_final, loss_type)

        # Backpropagate through layers (reverse order)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Gradient w.r.t. z (before activation)
            dA_dZ = self.activation_derivative(layer.z, layer.activation)
            dL_dZ = dL_dA * dA_dZ

            # Gradients w.r.t. weights and biases
            # dL/dW = dL/dZ @ A_prev^T
            layer.dW = dL_dZ @ layer.x.T / layer.x.shape[1]
            layer.db = np.sum(dL_dZ, axis=1, keepdims=True) / layer.x.shape[1]

            # Propagate gradient to previous layer
            if i > 0:
                dL_dA = layer.weights.T @ dL_dZ

        return loss

    def update_weights(self):
        """
        Update all weights using computed gradients.
        """
        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.biases -= self.learning_rate * layer.db

    def train_step(self, X, Y, loss_type='mse'):
        """
        Complete training step: forward + backward + update.

        Args:
            X: Input data
            Y: True labels
            loss_type: Loss function to use

        Returns:
            loss: Current loss value
        """
        # Forward pass
        predictions = self.forward(X)

        # Backward pass
        loss = self.backward(Y, loss_type)

        # Update weights
        self.update_weights()

        return loss

    def train(self, X, Y, epochs=1000, loss_type='mse', verbose=True):
        """
        Train the network for multiple epochs.

        Args:
            X: Training data (n_features, n_samples)
            Y: Training labels
            epochs: Number of training iterations
            loss_type: Loss function
            verbose: Whether to print progress

        Returns:
            history: List of loss values over training
        """
        history = []

        for epoch in range(epochs):
            loss = self.train_step(X, Y, loss_type)
            history.append(loss)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

        return history


# Example: XOR problem (classic non-linear problem)
def train_xor_network():
    """
    Train network to learn XOR function.

    XOR is not linearly separable, so needs hidden layer!
    """
    # XOR dataset
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])  # XOR outputs

    # Create network: 2 inputs -> 4 hidden (ReLU) -> 1 output (sigmoid)
    network = NeuralNetworkWithBackprop(
        layer_sizes=[2, 4, 1],
        activations=['relu', 'sigmoid'],
        learning_rate=0.1
    )

    print("Training XOR Network...")
    print("="*50)
    history = network.train(X, Y, epochs=5000, loss_type='mse')

    # Test predictions
    print("\n" + "="*50)
    print("Final Predictions:")
    predictions = network.forward(X)
    for i in range(X.shape[1]):
        x1, x2 = X[:, i]
        pred = predictions[0, i]
        true = Y[0, i]
        print(f"Input: [{x1}, {x2}] | Pred: {pred:.4f} | True: {true}")

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Network Training')
    plt.grid(True)
    plt.yscale('log')
    plt.show()

# Run the example
train_xor_network()
```

### 3.9 Gradient Checking - Verifying Backprop

How do we know our backprop implementation is correct?

**Numerical gradient** (slow but accurate):

$$
\frac{\partial L}{\partial W} \approx \frac{L(W + \epsilon) - L(W - \epsilon)}{2\epsilon}
$$

```python
def gradient_check(network, X, Y, epsilon=1e-7):
    """
    Verify backpropagation implementation using numerical gradients.

    Args:
        network: Neural network instance
        X: Input data
        Y: True labels
        epsilon: Small perturbation for numerical gradient

    Returns:
        Boolean: True if gradients match
    """
    # Compute analytical gradients
    network.forward(X)
    network.backward(Y)

    # Check each layer
    for i, layer in enumerate(network.layers):
        print(f"\nChecking Layer {i+1} gradients...")

        # Check a few weights
        for row in range(min(2, layer.weights.shape[0])):
            for col in range(min(2, layer.weights.shape[1])):
                # Analytical gradient
                analytical = layer.dW[row, col]

                # Numerical gradient
                original = layer.weights[row, col]

                layer.weights[row, col] = original + epsilon
                loss_plus = network.compute_loss(Y, network.forward(X))[0]

                layer.weights[row, col] = original - epsilon
                loss_minus = network.compute_loss(Y, network.forward(X))[0]

                layer.weights[row, col] = original  # Restore

                numerical = (loss_plus - loss_minus) / (2 * epsilon)

                # Compare
                diff = abs(analytical - numerical)
                match = diff < 1e-5

                print(f"  W[{row},{col}]: Analytical={analytical:.8f}, "
                      f"Numerical={numerical:.8f}, Diff={diff:.8e} {'✓' if match else '✗'}")
```

### 3.10 Key Takeaways from Day 3

✅ **Backpropagation = Chain rule applied systematically**

- Compute gradients layer by layer, backward
- Each layer's gradient depends on next layer's gradient

✅ **Loss function measures prediction quality**

- MSE for regression
- Cross-entropy for classification

✅ **Gradient descent updates weights**

- $W^{new} = W^{old} - \alpha \nabla L$
- Learning rate controls step size

✅ **Activation derivatives are crucial**

- ReLU: Simple (0 or 1)
- Sigmoid: $\sigma(z)(1-\sigma(z))$
- Must be differentiable!

✅ **Matrix form enables batch processing**

- Compute gradients for all samples at once
- Efficient and stable

---

<a name="day-4"></a>

## 📅 Day 4: Building Your First Complete Network

Today we put everything together and build a **production-ready** neural network from scratch!

### 4.1 Loss Functions Deep Dive

Let's implement all major loss functions with proper numerical stability.

```python
import numpy as np

class LossFunctions:
    """
    Collection of loss functions with stable implementations.

    Each function returns (loss, gradient) for backpropagation.
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error for regression.

        Formula: L = (1/n) Σ(y_true - y_pred)²

        Args:
            y_true: True values (n_outputs, batch_size)
            y_pred: Predictions (n_outputs, batch_size)

        Returns:
            loss (float): Scalar loss value
            gradient (array): ∂L/∂y_pred
        """
        batch_size = y_true.shape[1]

        # Loss
        loss = np.mean((y_pred - y_true) ** 2)

        # Gradient: ∂L/∂ŷ = 2(ŷ - y) / n
        gradient = 2 * (y_pred - y_true) / batch_size

        return loss, gradient

    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error for regression.

        Formula: L = (1/n) Σ|y_true - y_pred|
        More robust to outliers than MSE.

        Args:
            y_true: True values
            y_pred: Predictions

        Returns:
            loss, gradient
        """
        batch_size = y_true.shape[1]

        # Loss
        loss = np.mean(np.abs(y_pred - y_true))

        # Gradient: sign(ŷ - y)
        gradient = np.sign(y_pred - y_true) / batch_size

        return loss, gradient

    @staticmethod
    def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
        """
        Binary cross-entropy for binary classification.

        Formula: L = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

        Use with sigmoid activation in output layer.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
            epsilon: Clipping value to avoid log(0)

        Returns:
            loss, gradient
        """
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        batch_size = y_true.shape[1]

        # Loss
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        # Gradient: (ŷ - y) / [ŷ(1-ŷ)]
        # But with sigmoid output, this simplifies to: ŷ - y
        gradient = (y_pred - y_true) / batch_size

        return loss, gradient

    @staticmethod
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
        """
        Categorical cross-entropy for multi-class classification.

        Formula: L = -(1/n) Σ Σ y_c · log(ŷ_c)

        Use with softmax activation in output layer.

        Args:
            y_true: True labels, one-hot encoded (n_classes, batch_size)
            y_pred: Predicted probabilities (n_classes, batch_size)
            epsilon: Clipping value

        Returns:
            loss, gradient
        """
        # Clip predictions
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        batch_size = y_true.shape[1]

        # Loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

        # Gradient: With softmax + cross-entropy, gradient simplifies to: ŷ - y
        gradient = (y_pred - y_true) / batch_size

        return loss, gradient


# Test loss functions
def test_loss_functions():
    """Verify loss function implementations."""
    print("Testing Loss Functions")
    print("=" * 60)

    # Test MSE
    y_true = np.array([[1.0, 2.0, 3.0]])
    y_pred = np.array([[1.1, 1.9, 3.2]])

    loss, grad = LossFunctions.mse(y_true, y_pred)
    print(f"\nMSE Test:")
    print(f"  True: {y_true.flatten()}")
    print(f"  Pred: {y_pred.flatten()}")
    print(f"  Loss: {loss:.6f}")
    print(f"  Gradient: {grad.flatten()}")

    # Test Binary Cross-Entropy
    y_true = np.array([[1, 0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.8, 0.2]])

    loss, grad = LossFunctions.binary_crossentropy(y_true, y_pred)
    print(f"\nBinary Cross-Entropy Test:")
    print(f"  True: {y_true.flatten()}")
    print(f"  Pred: {y_pred.flatten()}")
    print(f"  Loss: {loss:.6f}")
    print(f"  Gradient: {grad.flatten()}")

    # Test Categorical Cross-Entropy
    y_true = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]).T  # One-hot encoded
    y_pred = np.array([[0.8, 0.1, 0.1],
                       [0.1, 0.7, 0.2],
                       [0.1, 0.2, 0.7]]).T

    loss, grad = LossFunctions.categorical_crossentropy(y_true, y_pred)
    print(f"\nCategorical Cross-Entropy Test:")
    print(f"  Loss: {loss:.6f}")
    print(f"  Average gradient magnitude: {np.mean(np.abs(grad)):.6f}")

test_loss_functions()
```

### 4.2 Optimizers - Beyond Basic Gradient Descent

Gradient descent has variants that improve training:

#### 4.2.1 SGD with Momentum

**Problem:** Basic SGD oscillates in ravines (valleys with steep sides).

**Solution:** Add momentum - keep moving in previous direction.

$$
v_t = \beta v_{t-1} + (1-\beta)\nabla L
$$

$$
W_t = W_{t-1} - \alpha v_t
$$

**Intuition:** Like a ball rolling downhill - accumulates velocity.

```python
class SGDMomentum:
    """
    Stochastic Gradient Descent with Momentum.

    Momentum helps accelerate SGD in relevant direction and
    dampens oscillations.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Args:
            learning_rate: Step size
            momentum: Momentum coefficient (typically 0.9)
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads, param_name):
        """
        Update parameters using momentum.

        Args:
            params: Current parameter values
            grads: Gradients
            param_name: Identifier for velocity tracking

        Returns:
            Updated parameters
        """
        # Initialize velocity if first time
        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(params)

        # Update velocity: v = β·v + (1-β)·∇L
        self.velocity[param_name] = (
            self.momentum * self.velocity[param_name] +
            (1 - self.momentum) * grads
        )

        # Update parameters: W = W - α·v
        params -= self.lr * self.velocity[param_name]

        return params
```

#### 4.2.2 RMSprop

**Problem:** Learning rate is same for all parameters.

**Solution:** Adapt learning rate per parameter based on gradient history.

$$
s_t = \beta s_{t-1} + (1-\beta)(\nabla L)^2
$$

$$
W_t = W_{t-1} - \frac{\alpha}{\sqrt{s_t + \epsilon}}\nabla L
$$

```python
class RMSprop:
    """
    RMSprop optimizer.

    Adapts learning rate for each parameter using moving average
    of squared gradients.
    """

    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """
        Args:
            learning_rate: Initial learning rate
            decay_rate: Decay rate for moving average (typically 0.9)
            epsilon: Small constant for numerical stability
        """
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads, param_name):
        """Update parameters using RMSprop."""
        # Initialize cache if first time
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(params)

        # Update cache: s = β·s + (1-β)·∇L²
        self.cache[param_name] = (
            self.decay_rate * self.cache[param_name] +
            (1 - self.decay_rate) * grads**2
        )

        # Update parameters: W = W - α·∇L / √(s + ε)
        params -= self.lr * grads / (np.sqrt(self.cache[param_name]) + self.epsilon)

        return params
```

#### 4.2.3 Adam (Adaptive Moment Estimation) ⭐ Most Popular!

**Combines** momentum + RMSprop.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L \quad \text{(momentum)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2 \quad \text{(RMSprop)}
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}
$$

$$
W_t = W_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
$$

```python
class Adam:
    """
    Adam optimizer - the default choice for most problems.

    Combines momentum and adaptive learning rates.
    Includes bias correction for early training steps.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate: Step size (typically 0.001)
            beta1: Decay rate for first moment (typically 0.9)
            beta2: Decay rate for second moment (typically 0.999)
            epsilon: Small constant for stability
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSprop)
        self.t = 0   # Time step

    def update(self, params, grads, param_name):
        """Update parameters using Adam."""
        # Initialize moments if first time
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(params)
            self.v[param_name] = np.zeros_like(params)

        self.t += 1

        # Update biased first moment estimate
        self.m[param_name] = (
            self.beta1 * self.m[param_name] +
            (1 - self.beta1) * grads
        )

        # Update biased second moment estimate
        self.v[param_name] = (
            self.beta2 * self.v[param_name] +
            (1 - self.beta2) * grads**2
        )

        # Bias correction
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        # Update parameters
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
```

### 4.3 Complete Neural Network Class (Production-Ready)

Now let's build the ultimate neural network class:

```python
class NeuralNetwork:
    """
    Complete production-ready neural network implementation.

    Features:
    - Multiple optimizers (SGD, Momentum, RMSprop, Adam)
    - Various loss functions
    - Batch training with shuffling
    - Training history tracking
    - Model saving/loading
    - Prediction methods
    """

    def __init__(self, layer_sizes, activations, optimizer='adam',
                 learning_rate=0.001, loss='mse'):
        """
        Initialize neural network.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activations: List of activation functions for each layer
            optimizer: 'sgd', 'momentum', 'rmsprop', or 'adam'
            learning_rate: Learning rate for optimizer
            loss: Loss function ('mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy')
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_fn = loss

        # Initialize layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = self._create_layer(layer_sizes[i], layer_sizes[i+1],
                                       activations[i])
            self.layers.append(layer)

        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate)

        # Training history
        self.history = {
            'loss': [],
            'val_loss': []
        }

        print(f"Created Neural Network:")
        print(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
        print(f"  Activations: {activations}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Loss: {loss}")

    def _create_layer(self, n_input, n_output, activation):
        """Create a layer with proper weight initialization."""
        layer = {
            'activation': activation,
            'input_size': n_input,
            'output_size': n_output
        }

        # He initialization for ReLU, Xavier for others
        if activation == 'relu':
            layer['W'] = np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)
        else:
            layer['W'] = np.random.randn(n_output, n_input) * np.sqrt(1.0 / n_input)

        layer['b'] = np.zeros((n_output, 1))

        return layer

    def _create_optimizer(self, optimizer_name, learning_rate):
        """Create optimizer instance."""
        if optimizer_name == 'sgd':
            return {'type': 'sgd', 'lr': learning_rate}
        elif optimizer_name == 'momentum':
            return SGDMomentum(learning_rate)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate)
        elif optimizer_name == 'adam':
            return Adam(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _activate(self, z, activation):
        """Apply activation function."""
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:  # linear
            return z

    def _activation_derivative(self, z, activation):
        """Compute activation derivative."""
        if activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:  # linear
            return np.ones_like(z)

    def forward(self, X):
        """
        Forward propagation through network.

        Args:
            X: Input data (n_features, batch_size)

        Returns:
            Final output and cache of intermediate values
        """
        cache = {'A0': X}
        A = X

        for i, layer in enumerate(self.layers):
            # Linear transformation
            Z = layer['W'] @ A + layer['b']

            # Activation
            A = self._activate(Z, layer['activation'])

            # Store for backprop
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A

        return A, cache

    def compute_loss(self, y_true, y_pred):
        """Compute loss and its gradient."""
        if self.loss_fn == 'mse':
            return LossFunctions.mse(y_true, y_pred)
        elif self.loss_fn == 'mae':
            return LossFunctions.mae(y_true, y_pred)
        elif self.loss_fn == 'binary_crossentropy':
            return LossFunctions.binary_crossentropy(y_true, y_pred)
        elif self.loss_fn == 'categorical_crossentropy':
            return LossFunctions.categorical_crossentropy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def backward(self, y_true, cache):
        """
        Backward propagation - compute all gradients.

        Args:
            y_true: True labels
            cache: Cached values from forward pass

        Returns:
            Dictionary of gradients
        """
        gradients = {}
        L = len(self.layers)

        # Get final output
        A_final = cache[f'A{L}']

        # Compute loss gradient
        loss, dA = self.compute_loss(y_true, A_final)

        # Backpropagate through layers
        for i in reversed(range(L)):
            layer = self.layers[i]

            # Get cached values
            Z = cache[f'Z{i+1}']
            A_prev = cache[f'A{i}']

            # Gradient w.r.t. Z
            dZ = dA * self._activation_derivative(Z, layer['activation'])

            # Gradients w.r.t. weights and biases
            batch_size = A_prev.shape[1]
            gradients[f'dW{i}'] = (dZ @ A_prev.T) / batch_size
            gradients[f'db{i}'] = np.sum(dZ, axis=1, keepdims=True) / batch_size

            # Gradient for previous layer
            if i > 0:
                dA = layer['W'].T @ dZ

        return loss, gradients

    def update_parameters(self, gradients):
        """Update all parameters using optimizer."""
        for i, layer in enumerate(self.layers):
            dW = gradients[f'dW{i}']
            db = gradients[f'db{i}']

            if isinstance(self.optimizer, dict):
                # Simple SGD
                layer['W'] -= self.optimizer['lr'] * dW
                layer['b'] -= self.optimizer['lr'] * db
            else:
                # Advanced optimizer
                layer['W'] = self.optimizer.update(layer['W'], dW, f'W{i}')
                layer['b'] = self.optimizer.update(layer['b'], db, f'b{i}')

    def train_step(self, X_batch, y_batch):
        """Single training step."""
        # Forward pass
        y_pred, cache = self.forward(X_batch)

        # Backward pass
        loss, gradients = self.backward(y_batch, cache)

        # Update parameters
        self.update_parameters(gradients)

        return loss

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network.

        Args:
            X_train: Training data (n_features, n_samples)
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            verbose: Whether to print progress

        Returns:
            Training history
        """
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"\nTraining for {epochs} epochs...")
        print(f"Samples: {n_samples} | Batch size: {batch_size} | Batches per epoch: {n_batches}")
        print("=" * 70)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]

            # Mini-batch training
            epoch_losses = []
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]

                loss = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)

            # Average loss for epoch
            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)

            # Validation loss
            if X_val is not None and y_val is not None:
                val_pred, _ = self.forward(X_val)
                val_loss, _ = self.compute_loss(y_val, val_pred)
                self.history['val_loss'].append(val_loss)

            # Print progress
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

        print("=" * 70)
        print("Training complete!")

        return self.history

    def predict(self, X):
        """Make predictions on new data."""
        y_pred, _ = self.forward(X)
        return y_pred

    def evaluate(self, X, y):
        """Evaluate model on test data."""
        y_pred = self.predict(X)
        loss, _ = self.compute_loss(y, y_pred)
        return loss


# Example: Train a network on synthetic data
def demo_complete_network():
    """
    Demonstrate complete network on regression problem.
    """
    print("\n" + "="*70)
    print("COMPLETE NEURAL NETWORK DEMO")
    print("="*70)

    # Generate synthetic data: y = sin(x)
    np.random.seed(42)
    X_train = np.random.uniform(-np.pi, np.pi, (1, 1000))
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)

    X_val = np.random.uniform(-np.pi, np.pi, (1, 200))
    y_val = np.sin(X_val) + np.random.normal(0, 0.1, X_val.shape)

    # Create network
    model = NeuralNetwork(
        layer_sizes=[1, 32, 32, 1],
        activations=['relu', 'relu', 'linear'],
        optimizer='adam',
        learning_rate=0.01,
        loss='mse'
    )

    # Train
    history = model.fit(X_train, y_train, X_val, y_val,
                       epochs=100, batch_size=32)

    # Visualize results
    import matplotlib.pyplot as plt

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Plot predictions
    X_test = np.linspace(-np.pi, np.pi, 100).reshape(1, -1)
    y_test = np.sin(X_test)
    y_pred = model.predict(X_test)

    plt.subplot(1, 3, 2)
    plt.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, s=10, label='Training Data')
    plt.plot(X_test.flatten(), y_test.flatten(), 'g-', linewidth=2, label='True Function')
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', linewidth=2, label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True)

    # Plot errors
    plt.subplot(1, 3, 3)
    errors = np.abs(y_pred - y_test)
    plt.plot(X_test.flatten(), errors.flatten())
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Errors')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('day4_complete_network.png', dpi=150)
    plt.show()

    print(f"\nFinal Training Loss: {history['loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")

# Run demo
demo_complete_network()
```

### 4.4 Debugging Neural Networks

Common issues and how to fix them:

#### Issue 1: Loss is NaN

**Causes:**

- Learning rate too high
- Numerical overflow in exp()
- Division by zero

**Solutions:**

```python
# Clip gradients
def clip_gradients(gradients, max_norm=5.0):
    """Prevent gradient explosion."""
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        for key in gradients:
            gradients[key] *= max_norm / total_norm

    return gradients
```

#### Issue 2: Loss Not Decreasing

**Causes:**

- Learning rate too small
- Bad weight initialization
- Wrong loss function

**Debug checklist:**

```python
def debug_training(model, X, y):
    """
    Diagnostic checks for training issues.
    """
    print("TRAINING DIAGNOSTICS")
    print("=" * 60)

    # 1. Check data
    print("\n1. Data Check:")
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Input range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"   Has NaN: {np.isnan(X).any() or np.isnan(y).any()}")

    # 2. Check forward pass
    y_pred, cache = model.forward(X)
    print(f"\n2. Forward Pass:")
    print(f"   Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"   Has NaN: {np.isnan(y_pred).any()}")

    # 3. Check gradients
    loss, grads = model.backward(y, cache)
    print(f"\n3. Gradients:")
    print(f"   Loss: {loss:.6f}")
    for key, grad in grads.items():
        print(f"   {key}: mean={np.mean(np.abs(grad)):.6f}, "
              f"max={np.max(np.abs(grad)):.6f}")

    # 4. Check weights
    print(f"\n4. Weights:")
    for i, layer in enumerate(model.layers):
        W = layer['W']
        print(f"   Layer {i}: mean={np.mean(np.abs(W)):.6f}, "
              f"std={np.std(W):.6f}")

    print("\n" + "=" * 60)
```

#### Issue 3: Overfitting

**Signs:**

- Training loss low, validation loss high
- Large gap between train and val loss

**Solutions:**

```python
class RegularizedLayer:
    """Layer with L2 regularization (weight decay)."""

    def __init__(self, n_input, n_output, activation, lambda_reg=0.01):
        self.W = np.random.randn(n_output, n_input) * np.sqrt(2.0/n_input)
        self.b = np.zeros((n_output, 1))
        self.activation = activation
        self.lambda_reg = lambda_reg

    def compute_regularization_loss(self):
        """Compute L2 regularization term."""
        return 0.5 * self.lambda_reg * np.sum(self.W ** 2)

    def compute_regularization_gradient(self):
        """Gradient of regularization term."""
        return self.lambda_reg * self.W


def add_dropout(A, dropout_rate=0.5, training=True):
    """
    Dropout regularization.

    Randomly set neurons to 0 during training.
    """
    if training and dropout_rate > 0:
        mask = np.random.rand(*A.shape) > dropout_rate
        A = A * mask / (1 - dropout_rate)  # Scale to maintain expected value
        return A, mask
    return A, None
```

### 4.5 Key Takeaways from Day 4

✅ **Loss functions must match the problem**

- Regression → MSE or MAE
- Binary classification → Binary cross-entropy
- Multi-class → Categorical cross-entropy

✅ **Optimizers improve training speed**

- SGD: Basic but slow
- Momentum: Better for ravines
- RMSprop: Adaptive per parameter
- Adam: Best default choice

✅ **Production code needs error handling**

- Gradient clipping
- Numerical stability (clip, epsilon)
- NaN/Inf checks

✅ **Debugging is essential**

- Check data ranges
- Verify gradients
- Monitor weight magnitudes

---

<a name="day-5"></a>

## 📅 Day 5: Introduction to PyTorch

Today we learn PyTorch - the industry-standard deep learning framework!

### 5.1 Why PyTorch?

**What we built so far:** Educational, from scratch
**What we need for real work:** Fast, GPU-accelerated, production-ready

**PyTorch advantages:**

- ✅ Automatic differentiation (no manual backprop!)
- ✅ GPU acceleration (100x faster)
- ✅ Pre-built layers and optimizers
- ✅ Huge ecosystem (pre-trained models, tools)
- ✅ Industry standard (research & production)

### 5.2 PyTorch Basics

#### Installation

```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Tensors - PyTorch's NumPy

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Creating tensors
print("Creating Tensors")
print("=" * 60)

# From Python list
t1 = torch.tensor([1, 2, 3])
print(f"From list: {t1}")

# From NumPy
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)
print(f"From numpy: {t2}")

# Random tensors
t3 = torch.randn(3, 4)  # Shape (3, 4), values from N(0,1)
print(f"Random tensor:\n{t3}")

# Zeros and ones
t4 = torch.zeros(2, 3)
t5 = torch.ones(2, 3)
print(f"Zeros:\n{t4}")
print(f"Ones:\n{t5}")

# Operations (similar to NumPy)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\nOperations:")
print(f"  Addition: {a + b}")
print(f"  Multiplication: {a * b}")
print(f"  Dot product: {torch.dot(a, b)}")
print(f"  Matrix multiply: {torch.mm(a.unsqueeze(0).T, b.unsqueeze(0))}")

# Moving to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

t_gpu = t3.to(device)
print(f"Tensor on GPU: {t_gpu.device}")
```

#### Automatic Differentiation - The Magic!

```python
def demo_autograd():
    """
    Demonstrate PyTorch's automatic differentiation.

    This is what makes PyTorch powerful!
    """
    print("\nAutomatic Differentiation Demo")
    print("=" * 60)

    # Create tensor with gradient tracking
    x = torch.tensor([2.0], requires_grad=True)

    # Define computation
    y = x ** 2 + 3 * x + 1

    print(f"x = {x.item()}")
    print(f"y = x² + 3x + 1 = {y.item()}")

    # Compute gradient automatically!
    y.backward()

    # dy/dx = 2x + 3 = 2(2) + 3 = 7
    print(f"dy/dx = {x.grad.item()}")
    print(f"Expected: 2x + 3 = {2*x.item() + 3}")

    # More complex example
    print("\nComplex Example:")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.sum(x ** 2)  # y = x₁² + x₂² + x₃²

    y.backward()

    # dy/dx_i = 2x_i
    print(f"x = {x.detach().numpy()}")
    print(f"y = Σx² = {y.item()}")
    print(f"dy/dx = {x.grad.numpy()}")
    print(f"Expected: 2x = {2 * x.detach().numpy()}")

demo_autograd()
```

### 5.3 Building Neural Networks in PyTorch

#### Method 1: Sequential API (Simple)

```python
# Simple sequential model
model_simple = nn.Sequential(
    nn.Linear(10, 20),   # Input: 10, Output: 20
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

print("Simple Sequential Model:")
print(model_simple)

# Forward pass
x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
output = model_simple(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

#### Method 2: Custom Module (Flexible) ⭐

```python
class NeuralNetwork(nn.Module):
    """
    Custom neural network class.

    This is the standard way to define networks in PyTorch.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize network architecture.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of outputs
        """
        super(NeuralNetwork, self).__init__()

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Combine into sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)
        """
        return self.network(x)


# Create model
model = NeuralNetwork(
    input_size=10,
    hidden_sizes=[32, 32, 16],
    output_size=1
)

print("\nCustom Neural Network:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
```

### 5.4 Training Loop in PyTorch

Here's the standard PyTorch training pattern:

```python
def train_pytorch_model():
    """
    Complete training example in PyTorch.

    This is the pattern you'll use for every project!
    """
    print("\n" + "="*70)
    print("PYTORCH TRAINING DEMO")
    print("="*70)

    # 1. Generate synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(1000, 10)
    y_train = torch.sum(X_train ** 2, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1

    X_val = torch.randn(200, 10)
    y_val = torch.sum(X_val ** 2, dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")

    # 2. Create model
    model = NeuralNetwork(
        input_size=10,
        hidden_sizes=[32, 32],
        output_size=1
    )

    # 3. Define loss function
    criterion = nn.MSELoss()

    # 4. Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training loop
    epochs = 100
    batch_size = 32
    history = {'train_loss': [], 'val_loss': []}

    print(f"\nTraining for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()  # Set to training mode

        # Mini-batch training
        train_losses = []
        for i in range(0, len(X_train), batch_size):
            # Get batch
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Don't compute gradients
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)
            history['val_loss'].append(val_loss.item())

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss.item():.6f}")

    print("=" * 70)
    print("Training complete!")

    # Visualize
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PyTorch Training History')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('day5_pytorch_training.png', dpi=150)
    plt.show()

    return model, history

# Train the model
trained_model, history = train_pytorch_model()
```

### 5.5 Common PyTorch Layers

```python
def explore_pytorch_layers():
    """
    Overview of commonly used PyTorch layers.
    """
    print("\nPyTorch Layers Overview")
    print("=" * 70)

    # Linear (Fully Connected)
    linear = nn.Linear(in_features=10, out_features=5)
    x = torch.randn(32, 10)  # Batch of 32
    out = linear(x)
    print(f"\n1. Linear Layer:")
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in linear.parameters())}")

    # Dropout (Regularization)
    dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
    x = torch.ones(5, 10)
    out_train = dropout(x)  # Training mode
    dropout.eval()
    out_eval = dropout(x)   # Evaluation mode
    print(f"\n2. Dropout Layer (p=0.5):")
    print(f"   Training mode (some zeros):\n{out_train[0, :5]}")
    print(f"   Eval mode (no dropout):\n{out_eval[0, :5]}")

    # BatchNorm (Normalization)
    batchnorm = nn.BatchNorm1d(num_features=10)
    x = torch.randn(32, 10)
    out = batchnorm(x)
    print(f"\n3. Batch Normalization:")
    print(f"   Input mean: {x.mean(dim=0)[0]:.4f}, std: {x.std(dim=0)[0]:.4f}")
    print(f"   Output mean: {out.mean(dim=0)[0]:.4f}, std: {out.std(dim=0)[0]:.4f}")

    # Activation functions
    print(f"\n4. Activation Functions:")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"   Input: {x.numpy()}")
    print(f"   ReLU: {torch.relu(x).numpy()}")
    print(f"   Sigmoid: {torch.sigmoid(x).numpy()}")
    print(f"   Tanh: {torch.tanh(x).numpy()}")

explore_pytorch_layers()
```

### 5.6 Saving and Loading Models

```python
def save_and_load_model():
    """
    Demonstrate model persistence.
    """
    print("\nModel Saving and Loading")
    print("=" * 70)

    # Create a model
    model = NeuralNetwork(10, [32, 32], 1)

    # Method 1: Save entire model
    torch.save(model, 'model_complete.pth')
    loaded_model = torch.load('model_complete.pth')
    print("✓ Saved and loaded complete model")

    # Method 2: Save only state dict (recommended)
    torch.save(model.state_dict(), 'model_weights.pth')

    # To load, create model first, then load weights
    new_model = NeuralNetwork(10, [32, 32], 1)
    new_model.load_state_dict(torch.load('model_weights.pth'))
    print("✓ Saved and loaded model weights")

    # Verify they're the same
    x = torch.randn(5, 10)
    out1 = model(x)
    out2 = new_model(x)
    print(f"\nOutputs match: {torch.allclose(out1, out2)}")

save_and_load_model()
```

### 5.7 Converting Our Custom Network to PyTorch

Let's convert our Day 4 network to PyTorch:

```python
class OurNetworkInPyTorch(nn.Module):
    """
    Our custom network from Day 4, reimplemented in PyTorch.

    Compare this to our NumPy version - much simpler!
    """

    def __init__(self, layer_sizes, activations, dropout_rate=0.0):
        super(OurNetworkInPyTorch, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = activations
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Build layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            # Apply activation
            if self.activations[i] == 'relu':
                x = torch.relu(x)
            elif self.activations[i] == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activations[i] == 'tanh':
                x = torch.tanh(x)

            # Apply dropout
            if self.dropout is not None and self.training:
                x = self.dropout(x)

        # Output layer
        x = self.layers[-1](x)
        if self.activations[-1] == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activations[-1] == 'softmax':
            x = torch.softmax(x, dim=1)

        return x


def compare_implementations():
    """
    Compare our NumPy implementation with PyTorch.
    """
    print("\nComparing Implementations")
    print("=" * 70)

    # Same architecture
    layer_sizes = [10, 32, 32, 1]
    activations = ['relu', 'relu', 'linear']

    # PyTorch model
    pytorch_model = OurNetworkInPyTorch(layer_sizes, activations)

    # Test forward pass
    x = torch.randn(64, 10)

    import time

    # PyTorch timing
    start = time.time()
    for _ in range(100):
        out = pytorch_model(x)
    pytorch_time = time.time() - start

    print(f"\nPyTorch:")
    print(f"  Time for 100 forward passes: {pytorch_time:.4f}s")
    print(f"  Output shape: {out.shape}")

    # NumPy timing (would be much slower)
    print(f"\nNumPy:")
    print(f"  Would be ~10-100x slower")
    print(f"  No GPU acceleration")
    print(f"  Manual backpropagation")

    print(f"\nPyTorch Advantages:")
    print(f"  ✓ Automatic differentiation")
    print(f"  ✓ GPU acceleration")
    print(f"  ✓ Built-in optimizers")
    print(f"  ✓ Cleaner code")
    print(f"  ✓ Production-ready")

compare_implementations()
```

### 5.8 Key Takeaways from Day 5

✅ **PyTorch is NumPy + Autograd + GPU**

- Tensors work like NumPy arrays
- Automatic differentiation (.backward())
- GPU acceleration with .to(device)

✅ **Standard training pattern**

```python
for epoch in epochs:
    optimizer.zero_grad()  # Clear gradients
    output = model(input)  # Forward pass
    loss = criterion(output, target)  # Compute loss
    loss.backward()        # Backward pass
    optimizer.step()       # Update weights
```

✅ **Two ways to define models**

- Sequential: Simple, for straightforward architectures
- Custom Module: Flexible, for complex architectures

✅ **Essential components**

- `nn.Module`: Base class for models
- `nn.Linear`: Fully connected layer
- `optim.Adam`: Optimizer
- `nn.MSELoss`, `nn.CrossEntropyLoss`: Loss functions

✅ **Training vs Evaluation mode**

- `model.train()`: Enables dropout, batch norm updates
- `model.eval()`: Disables dropout, uses running stats for BN
- `with torch.no_grad()`: Don't compute gradients (faster inference)

---

<a name="weekend-project"></a>

## 🎯 Weekend Project: MNIST Digit Recognition

Now let's build a complete project: recognizing handwritten digits!

### Project Overview

**Dataset:** MNIST (70,000 handwritten digits 0-9)
**Task:** Classify images into 10 classes
**Architecture:** Multi-layer neural network
**Framework:** PyTorch

### Step 1: Load and Explore Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    """
    print("Loading MNIST Dataset")
    print("=" * 70)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (0-1 range)
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize (mean, std)
    ])

    # Download and load data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: 10")

    return train_loader, test_loader, train_dataset, test_dataset

train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
```

### Step 2: Visualize the Data

```python
def visualize_mnist_samples(dataset, n_samples=10):
    """
    Visualize random MNIST samples.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(n_samples):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]

        # Convert to numpy and denormalize
        img = image.squeeze().numpy()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150)
    plt.show()

visualize_mnist_samples(train_dataset)
```

### Step 3: Build the Model

```python
class MNISTClassifier(nn.Module):
    """
    Neural network for MNIST digit classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden: 256 → 128 (with ReLU and Dropout)
    - Output: 10 (softmax probabilities)
    """

    def __init__(self, dropout_rate=0.2):
        super(MNISTClassifier, self).__init__()

        self.network = nn.Sequential(
            # Input layer
            nn.Flatten(),  # 28x28 → 784

            # Hidden layer 1
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Hidden layer 2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output layer
            nn.Linear(128, 10)
            # Note: No softmax here! CrossEntropyLoss includes it
        )

    def forward(self, x):
        return self.network(x)


# Create model
model = MNISTClassifier(dropout_rate=0.2)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {total_params:,}")
```

### Step 4: Training Function

```python
def train_mnist_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train MNIST classifier.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        Training history
    """
    # Move model to device
    model = model.to(device)

    # Loss function (includes softmax)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("\n" + "=" * 70)
    print("TRAINING MNIST CLASSIFIER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: Adam")
    print("=" * 70)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        # Testing phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        # Average test metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * test_correct / test_total

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)

        # Update learning rate
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

    print("=" * 70)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("Training complete!")

    return history


# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
history = train_mnist_model(model, train_loader, test_loader, epochs=10, device=device)
```

### Step 5: Visualize Results

```python
def plot_training_history(history):
    """
    Plot training and test metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('mnist_training_history.png', dpi=150)
    plt.show()

plot_training_history(history)
```

### Step 6: Test on Individual Samples

```python
def test_individual_predictions(model, test_dataset, n_samples=10, device='cpu'):
    """
    Test model on individual samples and visualize.
    """
    model.eval()
    model = model.to(device)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.ravel()

    for i in range(n_samples):
        # Get random sample
        idx = np.random.randint(len(test_dataset))
        image, true_label = test_dataset[idx]

        # Make prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
            output = model(image_batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_label = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_label].item()

        # Visualize
        img = image.squeeze().numpy()
        axes[i].imshow(img, cmap='gray')

        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {predicted_label}\n'
                         f'Confidence: {confidence:.2%}',
                         color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150)
    plt.show()

test_individual_predictions(model, test_dataset, device=device)
```

### Step 7: Confusion Matrix

```python
def plot_confusion_matrix(model, test_loader, device='cpu'):
    """
    Create confusion matrix for all test predictions.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            predictions = output.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(target.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - MNIST Classifier')
    plt.tight_layout()
    plt.savefig('mnist_confusion_matrix.png', dpi=150)
    plt.show()

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("=" * 40)
    for i in range(10):
        accuracy = 100 * cm[i, i] / cm[i, :].sum()
        print(f"Digit {i}: {accuracy:.2f}%")

plot_confusion_matrix(model, test_loader, device=device)
```

### Step 8: Save the Model

```python
def save_mnist_model(model, filepath='mnist_classifier.pth'):
    """
    Save trained model.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'dropout_rate': 0.2
        }
    }, filepath)
    print(f"\nModel saved to {filepath}")


def load_mnist_model(filepath='mnist_classifier.pth', device='cpu'):
    """
    Load trained model.
    """
    checkpoint = torch.load(filepath, map_location=device)

    model = MNISTClassifier(
        dropout_rate=checkpoint['model_architecture']['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {filepath}")
    return model

# Save model
save_mnist_model(model)

# Test loading
loaded_model = load_mnist_model(device=device)
```

### Weekend Project Summary

```python
print("\n" + "="*70)
print("WEEKEND PROJECT COMPLETE! 🎉")
print("="*70)
print("\nWhat You Built:")
print("  ✓ Complete MNIST digit classifier")
print("  ✓ ~98%+ accuracy on test set")
print("  ✓ Proper train/test split")
print("  ✓ Visualization of results")
print("  ✓ Confusion matrix analysis")
print("  ✓ Model saving/loading")
print("\nSkills Demonstrated:")
print("  ✓ PyTorch data loading")
print("  ✓ Model architecture design")
print("  ✓ Training loop implementation")
print("  ✓ Hyperparameter tuning")
print("  ✓ Model evaluation")
print("  ✓ Results visualization")
print("\nNext Steps:")
print("  • Try different architectures (more layers, different sizes)")
print("  • Experiment with hyperparameters (learning rate, dropout)")
print("  • Add data augmentation")
print("  • Try other datasets (Fashion-MNIST, CIFAR-10)")
print("="*70)
```

---

<a name="week-review"></a>

## 📚 Week 1 Review & Key Takeaways

Congratulations! You've completed Week 1. Let's review everything you've learned.

### What You've Mastered

#### Day 1: Single Neuron

- ✅ Neural networks are weighted sums + activation functions
- ✅ Activation functions introduce nonlinearity (ReLU, sigmoid, tanh)
- ✅ Weights determine importance, bias shifts decision boundary
- ✅ Single neuron creates linear decision boundary

#### Day 2: Forward Propagation

- ✅ Layers are collections of neurons working in parallel
- ✅ Matrix multiplication enables efficient computation
- ✅ Deep networks learn hierarchical features
- ✅ Forward pass is sequential layer application
- ✅ Batching improves efficiency and stability

#### Day 3: Backpropagation

- ✅ Backpropagation applies chain rule systematically
- ✅ Loss functions measure prediction quality
- ✅ Gradient descent updates weights to minimize loss
- ✅ Learning rate controls update step size
- ✅ Activation derivatives are crucial for backprop

#### Day 4: Complete Network

- ✅ Loss functions must match problem type
- ✅ Advanced optimizers (Momentum, RMSprop, Adam) improve training
- ✅ Production code needs error handling and debugging
- ✅ Regularization prevents overfitting

#### Day 5: PyTorch

- ✅ PyTorch provides automatic differentiation
- ✅ GPU acceleration dramatically speeds up training
- ✅ Standard training pattern: zero_grad → forward → loss → backward → step
- ✅ nn.Module is base class for all models
- ✅ train() vs eval() modes for dropout/batchnorm

#### Weekend: MNIST Project

- ✅ Complete end-to-end project workflow
- ✅ Data loading and preprocessing
- ✅ Model training and evaluation
- ✅ Results visualization
- ✅ Model persistence

### The Big Picture

```
NEURAL NETWORK = Function Approximator

Input → [Layer 1] → [Layer 2] → ... → [Layer L] → Output
         ↓          ↓                    ↓
      Features   Complex          Task-specific
                Patterns          Output

Learning = Adjusting weights to minimize loss

Forward Pass: Compute predictions
Backward Pass: Compute gradients
Weight Update: Move toward better predictions
```

### Mathematical Foundation

**Forward Pass:**

$$
\mathbf{a}^{[l]} = \sigma(\mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})
$$

**Loss:**

$$
L = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_i, \hat{y}_i)
$$

**Gradient Descent:**

$$
\mathbf{W} := \mathbf{W} - \alpha \frac{\partial L}{\partial \mathbf{W}}
$$

**Backpropagation (Chain Rule):**

$$
\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} \cdot ... \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}}
$$

### Code Patterns You've Learned

**NumPy (From Scratch):**

```python
# Forward
z = W @ x + b
a = activation(z)

# Backward
dz = da * activation_derivative(z)
dW = dz @ x.T
db = np.sum(dz)

# Update
W -= learning_rate * dW
b -= learning_rate * db
```

**PyTorch (Production):**

```python
# Forward
output = model(input)
loss = criterion(output, target)

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Common Pitfalls & Solutions

| Problem             | Cause                     | Solution                             |
| ------------------- | ------------------------- | ------------------------------------ |
| Loss is NaN         | Learning rate too high    | Reduce learning rate, clip gradients |
| Loss not decreasing | Learning rate too small   | Increase learning rate               |
| Training slow       | Using CPU                 | Use GPU with .to('cuda')             |
| Overfitting         | Model too complex         | Add dropout, reduce capacity         |
| Underfitting        | Model too simple          | Add layers, increase neurons         |
| Exploding gradients | Unstable training         | Gradient clipping, batch norm        |
| Vanishing gradients | Deep network with sigmoid | Use ReLU, residual connections       |

### Checklist: Can You...

- [ ] Explain what a neuron does mathematically?
- [ ] Implement forward propagation from scratch?
- [ ] Derive backpropagation equations?
- [ ] Explain the purpose of activation functions?
- [ ] Choose appropriate loss functions?
- [ ] Implement gradient descent variants?
- [ ] Build a neural network in PyTorch?
- [ ] Train a model on real data?
- [ ] Debug training issues?
- [ ] Visualize and evaluate results?

If you can check all these boxes, you're ready for Week 2! 🎉

### What's Next: Week 2 Preview

**Week 2: Training Deep Networks**

- Regularization techniques (L1, L2, Dropout)
- Batch normalization
- Advanced optimization (Adam, learning rate schedules)
- Hyperparameter tuning
- Better network architectures

### Resources for Further Practice

1. **Datasets to Try:**
   - Fashion-MNIST (harder than digits)
   - CIFAR-10 (color images)
   - Kaggle competitions

2. **Experiments to Run:**
   - Try different architectures
   - Compare optimizers
   - Tune hyperparameters
   - Add regularization

3. **Challenges:**
   - Beat 99% accuracy on MNIST
   - Implement custom activation function
   - Build network without PyTorch layers
   - Create visualization tool

### Final Words

You've built a **solid foundation** in neural networks:

- ✅ Understanding from first principles
- ✅ Mathematical rigor
- ✅ From-scratch implementation
- ✅ Production-ready code
- ✅ Complete project experience

This is **real knowledge** - not just surface-level tutorials. You can now:

- Read research papers and understand them
- Implement papers from scratch
- Debug training issues effectively
- Build production systems

**Keep building, keep learning!** 🚀

---

## 📖 Additional Resources

### Papers Referenced

1. Universal Approximation Theorem (Cybenko, 1989)
2. Backpropagation (Rumelhart et al., 1986)
3. Adam Optimizer (Kingma & Ba, 2014)
4. Dropout (Srivastava et al., 2014)

### Books

- "Deep Learning" by Goodfellow, Bengio, Courville (Chapter 6)
- "Neural Networks and Deep Learning" by Michael Nielsen

### Online Courses

- Andrew Ng's Deep Learning Specialization (Course 1)
- Fast.ai Practical Deep Learning for Coders

### Practice Platforms

- Kaggle (competitions and datasets)
- Google Colab (free GPU)
- PyTorch Tutorials (official docs)

---

**🎓 Week 1 Complete! Ready for Week 2!**
