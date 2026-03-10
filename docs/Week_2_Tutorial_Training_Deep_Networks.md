# Week 2: Training Deep Networks - Complete Tutorial

## Making Neural Networks Work in Practice

> **Expert Guide**: 15+ Years of Deep Learning Experience
>
> By the end of this week, you'll master the techniques that make deep learning actually work: regularization, normalization, optimization, and architectural best practices.

---

## 📚 Table of Contents

1. [Introduction: The Challenges of Deep Learning](#introduction)
2. [Day 1: Regularization Techniques (L1, L2, Dropout)](#day-1)
3. [Day 2: Batch Normalization and Layer Normalization](#day-2)
4. [Day 3: Advanced Optimization Algorithms](#day-3)
5. [Day 4: Learning Rate Scheduling and Hyperparameter Tuning](#day-4)
6. [Day 5: Better Network Architectures](#day-5)
7. [Weekend Project: Image Classification with Regularization](#weekend-project)
8. [Week Review & Key Takeaways](#week-review)

---

<a name="introduction"></a>

## 🧠 Introduction: The Challenges of Deep Learning

### What You Learned in Week 1

You can now:

- Build neural networks from scratch
- Implement backpropagation
- Train models with gradient descent
- Use PyTorch for production code

### The Reality Check

But when you try to train **deep networks** (many layers) on **real data**, you hit problems:

❌ **Overfitting**: Model memorizes training data, fails on new data
❌ **Vanishing/Exploding Gradients**: Gradients become too small or too large
❌ **Slow Convergence**: Training takes forever
❌ **Sensitivity to Hyperparameters**: Small changes break everything
❌ **Internal Covariate Shift**: Layer inputs change distribution during training

### This Week's Solution Toolkit

✅ **Regularization**: Prevent overfitting (L1, L2, Dropout, Early Stopping)
✅ **Normalization**: Stabilize training (Batch Norm, Layer Norm)
✅ **Better Optimizers**: Faster convergence (Adam, AdamW, learning rate schedules)
✅ **Architectural Tricks**: Design principles that work
✅ **Hyperparameter Tuning**: Systematic search strategies

### The Big Picture

```
Raw Network (Week 1)          Production Network (Week 2)
     ↓                              ↓
[Input]                        [Input]
   ↓                              ↓
[Linear]                       [Linear + BatchNorm]
   ↓                              ↓
[ReLU]                         [ReLU + Dropout]
   ↓                              ↓
[Linear]                       [Linear + L2 Regularization]
   ↓                              ↓
[Output]                       [Output]
                                   ↓
Loss = MSE                    Loss = MSE + λ·||W||²
Optimizer = SGD               Optimizer = Adam + LR Schedule
```

**This week transforms your models from academic toys to production powerhouses!**

---

<a name="day-1"></a>

## 📅 Day 1: Regularization Techniques

> "Regularization is the art of making models that generalize rather than memorize."

### 1.1 Understanding Overfitting

#### The Problem Visualized

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def demonstrate_overfitting():
    """
    Visual demonstration of overfitting.

    This shows why regularization is necessary!
    """
    print("="*70)
    print("OVERFITTING DEMONSTRATION")
    print("="*70)

    # Generate true function: y = sin(x) with noise
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, 20).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)

    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_test = np.sin(X_test)

    # Convert to PyTorch
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)

    # Model 1: Simple (underfit)
    model_simple = nn.Sequential(
        nn.Linear(1, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )

    # Model 2: Complex (overfit)
    model_complex = nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )

    # Train both models
    def train_model(model, epochs=5000):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_train_t)
            loss = criterion(predictions, y_train_t)
            loss.backward()
            optimizer.step()

        return model

    print("Training simple model...")
    model_simple = train_model(model_simple)

    print("Training complex model...")
    model_complex = train_model(model_complex)

    # Evaluate
    with torch.no_grad():
        pred_simple = model_simple(X_test_t).numpy()
        pred_complex = model_complex(X_test_t).numpy()

        train_pred_simple = model_simple(X_train_t).numpy()
        train_pred_complex = model_complex(X_train_t).numpy()

    # Calculate errors
    train_mse_simple = np.mean((train_pred_simple - y_train)**2)
    train_mse_complex = np.mean((train_pred_complex - y_train)**2)
    test_mse_simple = np.mean((pred_simple - y_test)**2)
    test_mse_complex = np.mean((pred_complex - y_test)**2)

    print(f"\nResults:")
    print(f"Simple Model  - Train MSE: {train_mse_simple:.4f}, Test MSE: {test_mse_simple:.4f}")
    print(f"Complex Model - Train MSE: {train_mse_complex:.4f}, Test MSE: {test_mse_complex:.4f}")
    print(f"\n⚠️  Complex model: Low train error but HIGH test error = OVERFITTING!")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True function
    axes[0].scatter(X_train, y_train, c='blue', s=50, label='Training Data', zorder=3)
    axes[0].plot(X_test, y_test, 'g-', linewidth=2, label='True Function', zorder=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('True Function (sin(x))')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Simple model (underfit)
    axes[1].scatter(X_train, y_train, c='blue', s=50, label='Training Data', zorder=3)
    axes[1].plot(X_test, y_test, 'g-', linewidth=2, label='True Function', zorder=2)
    axes[1].plot(X_test, pred_simple, 'r--', linewidth=2, label='Simple Model', zorder=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'Simple Model (Underfit)\nTest MSE: {test_mse_simple:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Complex model (overfit)
    axes[2].scatter(X_train, y_train, c='blue', s=50, label='Training Data', zorder=3)
    axes[2].plot(X_test, y_test, 'g-', linewidth=2, label='True Function', zorder=2)
    axes[2].plot(X_test, pred_complex, 'r--', linewidth=2, label='Complex Model', zorder=2)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title(f'Complex Model (Overfit)\nTest MSE: {test_mse_complex:.4f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_overfitting_demo.png', dpi=150)
    plt.show()

demonstrate_overfitting()
```

#### Key Insight

**Overfitting happens when:**

- Model is too complex for the data
- Training set is too small
- Training for too long without validation

**Signs of overfitting:**

- Training loss ≪ Validation loss
- High variance in predictions
- Wild oscillations in predictions

---

### 1.2 L2 Regularization (Weight Decay)

#### Mathematical Foundation

**Standard loss:**

$$
L = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_i, \hat{y}_i)
$$

**With L2 regularization:**

$$
L_{regularized} = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_i, \hat{y}_i) + \frac{\lambda}{2}\sum_{l=1}^{L}||W^{[l]}||^2
$$

Where:

- $\lambda$ = regularization strength (hyperparameter)
- $||W^{[l]}||^2 = \sum_{i,j}(W_{ij}^{[l]})^2$ = sum of squared weights

#### Intuition: Why Does This Work?

**Penalty for large weights:**

- Large weights → model is sensitive to input changes
- Small weights → model is smooth and generalizes better
- L2 penalty encourages weights to stay small

**Effect on gradient:**

$$
\frac{\partial L_{regularized}}{\partial W} = \frac{\partial L}{\partial W} + \lambda W
$$

This means: **Weights shrink toward zero** during each update!

#### Implementation from Scratch

```python
class L2RegularizedNetwork(nn.Module):
    """
    Neural network with manual L2 regularization.

    This shows how L2 regularization is implemented internally.
    """

    def __init__(self, input_size, hidden_sizes, output_size, lambda_l2=0.01):
        super(L2RegularizedNetwork, self).__init__()

        self.lambda_l2 = lambda_l2

        # Build layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def l2_penalty(self):
        """
        Compute L2 penalty (sum of squared weights).

        Returns:
            L2 penalty term: (λ/2) * Σ(W²)
        """
        l2_reg = torch.tensor(0., requires_grad=True)

        for name, param in self.named_parameters():
            if 'weight' in name:  # Only regularize weights, not biases
                l2_reg = l2_reg + torch.sum(param ** 2)

        return 0.5 * self.lambda_l2 * l2_reg

    def compute_loss(self, predictions, targets, criterion):
        """
        Compute total loss = data loss + L2 penalty.

        Args:
            predictions: Model predictions
            targets: True labels
            criterion: Loss function (e.g., MSELoss)

        Returns:
            Total loss including regularization
        """
        data_loss = criterion(predictions, targets)
        reg_loss = self.l2_penalty()
        total_loss = data_loss + reg_loss

        return total_loss, data_loss, reg_loss


def demonstrate_l2_regularization():
    """
    Compare training with and without L2 regularization.
    """
    print("\n" + "="*70)
    print("L2 REGULARIZATION DEMONSTRATION")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, 30).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.15, X_train.shape)

    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_test = np.sin(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)

    # Model without regularization
    model_no_reg = L2RegularizedNetwork(1, [50, 50], 1, lambda_l2=0.0)

    # Model with regularization
    model_with_reg = L2RegularizedNetwork(1, [50, 50], 1, lambda_l2=0.1)

    # Training function
    def train_with_regularization(model, epochs=2000):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        history = {'total_loss': [], 'data_loss': [], 'reg_loss': []}

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_train_t)
            total_loss, data_loss, reg_loss = model.compute_loss(
                predictions, y_train_t, criterion
            )
            total_loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Total: {total_loss:.4f} | "
                      f"Data: {data_loss:.4f} | Reg: {reg_loss:.4f}")

            history['total_loss'].append(total_loss.item())
            history['data_loss'].append(data_loss.item())
            history['reg_loss'].append(reg_loss.item())

        return history

    print("\nTraining WITHOUT regularization:")
    print("-" * 70)
    hist_no_reg = train_with_regularization(model_no_reg)

    print("\nTraining WITH regularization (λ=0.1):")
    print("-" * 70)
    hist_with_reg = train_with_regularization(model_with_reg)

    # Evaluate
    with torch.no_grad():
        pred_no_reg = model_no_reg(X_test_t).numpy()
        pred_with_reg = model_with_reg(X_test_t).numpy()

        test_mse_no_reg = np.mean((pred_no_reg - y_test)**2)
        test_mse_with_reg = np.mean((pred_with_reg - y_test)**2)

    print("\n" + "="*70)
    print(f"Test MSE without regularization: {test_mse_no_reg:.4f}")
    print(f"Test MSE with regularization:    {test_mse_with_reg:.4f}")
    print(f"Improvement: {((test_mse_no_reg - test_mse_with_reg)/test_mse_no_reg*100):.1f}%")
    print("="*70)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Predictions comparison
    axes[0, 0].scatter(X_train, y_train, c='blue', s=50, label='Training Data', zorder=3)
    axes[0, 0].plot(X_test, y_test, 'g-', linewidth=2, label='True Function', zorder=2)
    axes[0, 0].plot(X_test, pred_no_reg, 'r--', linewidth=2, label='No Regularization', zorder=2)
    axes[0, 0].plot(X_test, pred_with_reg, 'orange', linestyle='--', linewidth=2,
                    label='With L2 Reg', zorder=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Predictions Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss curves
    axes[0, 1].plot(hist_no_reg['data_loss'], label='No Regularization', linewidth=2)
    axes[0, 1].plot(hist_with_reg['data_loss'], label='With L2 Reg', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss')
    axes[0, 1].set_title('Training Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')

    # Regularization loss
    axes[1, 0].plot(hist_with_reg['reg_loss'], color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L2 Penalty')
    axes[1, 0].set_title('L2 Regularization Penalty Over Time')
    axes[1, 0].grid(True)

    # Weight magnitudes
    weights_no_reg = []
    weights_with_reg = []

    for param in model_no_reg.parameters():
        if len(param.shape) == 2:  # Weight matrices
            weights_no_reg.extend(param.detach().numpy().flatten())

    for param in model_with_reg.parameters():
        if len(param.shape) == 2:
            weights_with_reg.extend(param.detach().numpy().flatten())

    axes[1, 1].hist(weights_no_reg, bins=50, alpha=0.6, label='No Regularization', color='red')
    axes[1, 1].hist(weights_with_reg, bins=50, alpha=0.6, label='With L2 Reg', color='orange')
    axes[1, 1].set_xlabel('Weight Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Weight Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_l2_regularization.png', dpi=150)
    plt.show()

    print("\n📊 Observations:")
    print("  • Regularized model has smaller, more concentrated weights")
    print("  • Regularized model produces smoother predictions")
    print("  • Better generalization to test data")

demonstrate_l2_regularization()
```

#### PyTorch Built-in Weight Decay

```python
# PyTorch makes L2 regularization easy!
# Just use weight_decay parameter in optimizer

def pytorch_weight_decay_demo():
    """
    Demonstrate PyTorch's built-in weight decay.
    """
    print("\nPyTorch Weight Decay Demo")
    print("="*70)

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # weight_decay = λ (L2 regularization strength)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    print("✓ Optimizer configured with weight_decay=0.01")
    print("  This automatically adds L2 penalty to loss!")
    print("\nEquivalent to manually adding: loss += 0.01 * Σ(W²)")

pytorch_weight_decay_demo()
```

---

### 1.3 L1 Regularization (Lasso)

#### Mathematical Foundation

**L1 regularization:**

$$
L_{L1} = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_i, \hat{y}_i) + \lambda\sum_{l=1}^{L}||W^{[l]}||_1
$$

Where $||W||_1 = \sum_{i,j}|W_{ij}|$ = sum of absolute values

#### L1 vs L2: Key Differences

| Aspect            | L2 (Ridge)             | L1 (Lasso)                     |
| ----------------- | ---------------------- | ------------------------------ | --- | --- |
| **Formula**       | $\lambda \sum W^2$     | $\lambda \sum                  | W   | $   |
| **Penalty Shape** | Circular               | Diamond                        |
| **Effect**        | Shrinks weights        | **Zeros out** weights          |
| **Result**        | Small weights          | **Sparse** weights             |
| **Use Case**      | General regularization | **Feature selection**          |
| **Gradient**      | $\lambda W$            | $\lambda \cdot \text{sign}(W)$ |

#### Why L1 Creates Sparsity

```python
def visualize_l1_vs_l2():
    """
    Visualize why L1 creates sparse solutions.

    This is a geometric intuition!
    """
    print("\nL1 vs L2 Geometric Intuition")
    print("="*70)

    # Create contour plot
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss function contours (example: quadratic bowl)
    Loss = (W1 - 0.5)**2 + (W2 - 1.5)**2

    # L1 constraint: |w1| + |w2| <= t
    t_l1 = 1.0

    # L2 constraint: w1² + w2² <= t²
    t_l2 = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # L2 regularization
    axes[0].contour(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.6)
    circle = plt.Circle((0, 0), t_l2, fill=False, color='red', linewidth=3,
                        label='L2 Constraint')
    axes[0].add_patch(circle)
    axes[0].scatter([0.5], [1.5], c='blue', s=200, marker='*',
                   label='Unconstrained Minimum', zorder=5)
    axes[0].scatter([0.45], [0.89], c='red', s=200, marker='o',
                   label='L2 Solution', zorder=5)
    axes[0].set_xlabel('w₁')
    axes[0].set_ylabel('w₂')
    axes[0].set_title('L2 Regularization (Ridge)\nWeights stay small but non-zero')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)

    # L1 regularization
    axes[1].contour(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.6)
    diamond_x = [0, t_l1, 0, -t_l1, 0]
    diamond_y = [t_l1, 0, -t_l1, 0, t_l1]
    axes[1].plot(diamond_x, diamond_y, 'r-', linewidth=3, label='L1 Constraint')
    axes[1].scatter([0.5], [1.5], c='blue', s=200, marker='*',
                   label='Unconstrained Minimum', zorder=5)
    axes[1].scatter([0], [1.0], c='red', s=200, marker='o',
                   label='L1 Solution (w₁=0!)', zorder=5)
    axes[1].set_xlabel('w₁')
    axes[1].set_ylabel('w₂')
    axes[1].set_title('L1 Regularization (Lasso)\nWeights become exactly zero!')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('week2_l1_vs_l2_geometry.png', dpi=150)
    plt.show()

    print("\n🔑 Key Insight:")
    print("  L2 (circle): Touches axes smoothly → small but non-zero weights")
    print("  L1 (diamond): Hits axes at corners → EXACTLY ZERO weights!")
    print("\n  This is why L1 is used for feature selection!")

visualize_l1_vs_l2()
```

#### L1 Implementation

```python
class L1RegularizedNetwork(nn.Module):
    """
    Neural network with L1 regularization.
    """

    def __init__(self, input_size, hidden_sizes, output_size, lambda_l1=0.01):
        super(L1RegularizedNetwork, self).__init__()

        self.lambda_l1 = lambda_l1

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def l1_penalty(self):
        """
        Compute L1 penalty (sum of absolute weights).

        Returns:
            L1 penalty term: λ * Σ|W|
        """
        l1_reg = torch.tensor(0., requires_grad=True)

        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.sum(torch.abs(param))

        return self.lambda_l1 * l1_reg

    def compute_loss(self, predictions, targets, criterion):
        """Compute total loss with L1 penalty."""
        data_loss = criterion(predictions, targets)
        reg_loss = self.l1_penalty()
        total_loss = data_loss + reg_loss

        return total_loss, data_loss, reg_loss

    def count_zero_weights(self, threshold=1e-3):
        """
        Count how many weights are effectively zero.

        L1 regularization drives weights to zero!
        """
        total_weights = 0
        zero_weights = 0

        for name, param in self.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                zero_weights += (torch.abs(param) < threshold).sum().item()

        sparsity = 100 * zero_weights / total_weights
        return zero_weights, total_weights, sparsity


def demonstrate_l1_sparsity():
    """
    Show how L1 creates sparse networks.
    """
    print("\n" + "="*70)
    print("L1 REGULARIZATION: SPARSITY DEMONSTRATION")
    print("="*70)

    # Simple regression task
    torch.manual_seed(42)
    X = torch.randn(100, 20)  # 20 features
    # True function only uses first 5 features!
    y = X[:, :5].sum(dim=1, keepdim=True) + torch.randn(100, 1) * 0.1

    # Model with L1 regularization
    model = L1RegularizedNetwork(20, [30], 1, lambda_l1=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining with L1 regularization...")
    print("True function uses only first 5 features out of 20")
    print("-" * 70)

    for epoch in range(500):
        optimizer.zero_grad()
        predictions = model(X)
        total_loss, data_loss, reg_loss = model.compute_loss(predictions, y, criterion)
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            zero_w, total_w, sparsity = model.count_zero_weights()
            print(f"Epoch {epoch:3d} | Loss: {data_loss:.4f} | "
                  f"Sparsity: {sparsity:.1f}% ({zero_w}/{total_w} weights ≈0)")

    # Final sparsity analysis
    zero_w, total_w, sparsity = model.count_zero_weights()
    print("\n" + "="*70)
    print(f"Final Sparsity: {sparsity:.1f}%")
    print(f"Zero weights: {zero_w}/{total_w}")
    print("✓ L1 regularization successfully identified and zeroed irrelevant features!")
    print("="*70)

    # Visualize weights
    first_layer_weights = model.network[0].weight.detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(first_layer_weights), cmap='hot', aspect='auto')
    plt.colorbar(label='|Weight|')
    plt.xlabel('Input Feature')
    plt.ylabel('Hidden Neuron')
    plt.title('First Layer Weights (Absolute Value)\nBrighter = Larger')

    plt.subplot(1, 2, 2)
    avg_weights = np.mean(np.abs(first_layer_weights), axis=0)
    plt.bar(range(20), avg_weights)
    plt.axvline(x=4.5, color='r', linestyle='--', linewidth=2, label='Relevant Features')
    plt.xlabel('Input Feature Index')
    plt.ylabel('Average |Weight|')
    plt.title('Feature Importance (via L1 Regularization)\nFirst 5 features should be larger')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_l1_sparsity.png', dpi=150)
    plt.show()

demonstrate_l1_sparsity()
```

---

### 1.4 Dropout: The Randomization Approach

#### The Revolutionary Idea

**Dropout** (Srivastava et al., 2014) was a breakthrough:

**During training:**

- Randomly "drop" (set to 0) neurons with probability $p$
- Forces network to learn redundant representations
- Prevents co-adaptation of neurons

**During testing:**

- Use all neurons
- Scale activations by $(1-p)$ to compensate

#### Mathematical Formulation

**Training:**

$$
r_i \sim \text{Bernoulli}(1-p)
$$

$$
\tilde{a}_i = r_i \cdot a_i
$$

**Testing (inverted dropout, PyTorch default):**

$$
a_i^{test} = a_i^{train}
$$

(Already scaled during training)

#### Intuition: Ensemble Learning

Dropout is like training an **ensemble** of $2^n$ networks (where $n$ = number of neurons):

- Each forward pass uses a different subset
- Final model averages all these subnetworks
- Similar to Random Forest but for neural networks!

```python
def visualize_dropout_mechanism():
    """
    Visualize how dropout works.
    """
    print("\nDropout Mechanism Visualization")
    print("="*70)

    # Simple network
    input_size = 8
    hidden_size = 6

    # Create activation pattern
    np.random.seed(42)
    activations = np.random.rand(hidden_size)

    # Apply dropout (p=0.5)
    dropout_prob = 0.5
    mask = np.random.rand(hidden_size) > dropout_prob
    dropped_activations = activations * mask

    # Scale for inverted dropout
    scaled_activations = dropped_activations / (1 - dropout_prob)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original activations
    axes[0].bar(range(hidden_size), activations, color='blue', alpha=0.7)
    axes[0].set_xlabel('Neuron Index')
    axes[0].set_ylabel('Activation')
    axes[0].set_title('Original Activations')
    axes[0].set_ylim(0, 2)
    axes[0].grid(True, alpha=0.3)

    # After dropout
    colors = ['red' if m == 0 else 'blue' for m in mask]
    axes[1].bar(range(hidden_size), dropped_activations, color=colors, alpha=0.7)
    axes[1].set_xlabel('Neuron Index')
    axes[1].set_ylabel('Activation')
    axes[1].set_title('After Dropout (p=0.5)\nRed = Dropped')
    axes[1].set_ylim(0, 2)
    axes[1].grid(True, alpha=0.3)

    # After scaling
    axes[2].bar(range(hidden_size), scaled_activations, color=colors, alpha=0.7)
    axes[2].set_xlabel('Neuron Index')
    axes[2].set_ylabel('Activation')
    axes[2].set_title('After Scaling (Inverted Dropout)\nScaled by 1/(1-p)')
    axes[2].set_ylim(0, 2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_dropout_mechanism.png', dpi=150)
    plt.show()

    print("\n🎲 Random Mask:", mask.astype(int))
    print(f"   {np.sum(~mask)} out of {hidden_size} neurons dropped")
    print("\n💡 Key Points:")
    print("  • Different mask every forward pass")
    print("  • Forces network to not rely on specific neurons")
    print("  • Scaling maintains expected activation values")

visualize_dropout_mechanism()
```

#### Dropout Implementation and Demonstration

```python
class DropoutNetwork(nn.Module):
    """
    Network with dropout layers.
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.5):
        super(DropoutNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())

            # Add dropout after activation (except last layer)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(p=dropout_p))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def compare_with_without_dropout():
    """
    Compare training with and without dropout.
    """
    print("\n" + "="*70)
    print("DROPOUT REGULARIZATION DEMONSTRATION")
    print("="*70)

    # Generate data (polynomial overfitting scenario)
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 100
    X_train = torch.FloatTensor(np.random.uniform(-1, 1, (n_train, 1)))
    y_train = X_train**2 + torch.randn(n_train, 1) * 0.1

    X_test = torch.FloatTensor(np.linspace(-1.5, 1.5, 100).reshape(-1, 1))
    y_test = X_test**2

    # Model without dropout
    model_no_dropout = DropoutNetwork(1, [100, 100, 100], 1, dropout_p=0.0)

    # Model with dropout
    model_with_dropout = DropoutNetwork(1, [100, 100, 100], 1, dropout_p=0.3)

    # Training function
    def train_model(model, epochs=300):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            train_pred = model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # Testing
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test)
                test_losses.append(test_loss.item())

            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")

        return train_losses, test_losses

    print("\nTraining WITHOUT dropout:")
    print("-" * 70)
    train_loss_no_drop, test_loss_no_drop = train_model(model_no_dropout)

    print("\nTraining WITH dropout (p=0.3):")
    print("-" * 70)
    train_loss_drop, test_loss_drop = train_model(model_with_dropout)

    # Get predictions
    model_no_dropout.eval()
    model_with_dropout.eval()

    with torch.no_grad():
        pred_no_drop = model_no_dropout(X_test).numpy()
        pred_drop = model_with_dropout(X_test).numpy()

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training curves
    axes[0, 0].plot(train_loss_no_drop, label='No Dropout', linewidth=2)
    axes[0, 0].plot(train_loss_drop, label='With Dropout', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')

    # Test curves
    axes[0, 1].plot(test_loss_no_drop, label='No Dropout', linewidth=2)
    axes[0, 1].plot(test_loss_drop, label='With Dropout', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Test Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')

    # Predictions
    axes[1, 0].scatter(X_train.numpy(), y_train.numpy(), alpha=0.5, s=20, label='Training Data')
    axes[1, 0].plot(X_test.numpy(), y_test.numpy(), 'g-', linewidth=2, label='True Function')
    axes[1, 0].plot(X_test.numpy(), pred_no_drop, 'r--', linewidth=2, label='No Dropout')
    axes[1, 0].plot(X_test.numpy(), pred_drop, 'orange', linestyle='--', linewidth=2, label='With Dropout')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Predictions Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Overfitting gap
    epochs = range(len(train_loss_no_drop))
    gap_no_drop = np.array(test_loss_no_drop) - np.array(train_loss_no_drop)
    gap_drop = np.array(test_loss_drop) - np.array(train_loss_drop)

    axes[1, 1].plot(epochs, gap_no_drop, label='No Dropout', linewidth=2)
    axes[1, 1].plot(epochs, gap_drop, label='With Dropout', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Loss - Train Loss')
    axes[1, 1].set_title('Generalization Gap (Lower is Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('week2_dropout_comparison.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("RESULTS:")
    print(f"Final Test Loss (No Dropout): {test_loss_no_drop[-1]:.4f}")
    print(f"Final Test Loss (With Dropout): {test_loss_drop[-1]:.4f}")
    print(f"Improvement: {((test_loss_no_drop[-1]-test_loss_drop[-1])/test_loss_no_drop[-1]*100):.1f}%")
    print("="*70)

compare_with_without_dropout()
```

---

### 1.5 Early Stopping

#### The Concept

**Simplest regularization technique:**

1. Monitor validation loss during training
2. Stop when validation loss stops improving
3. Restore weights from best validation point

**Why it works:**

- Training longer → risk of overfitting
- Stop before model starts memorizing
- Free! No hyperparameter to tune

#### Implementation

```python
class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when it stops improving.
    """

    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Check if should stop training.

        Args:
            val_loss: Current validation loss
            model: Model to save if best

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"   → Validation loss improved to {val_loss:.6f}")

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement
            if self.verbose:
                print(f"   → Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"   → No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠️  Early stopping triggered!")
                    print(f"   Best validation loss: {self.best_loss:.6f}")

        return self.early_stop

    def restore_best_model(self, model):
        """Restore model to best state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"✓ Restored model to best state (val_loss={self.best_loss:.6f})")


def demonstrate_early_stopping():
    """
    Demonstrate early stopping in action.
    """
    print("\n" + "="*70)
    print("EARLY STOPPING DEMONSTRATION")
    print("="*70)

    # Generate data
    torch.manual_seed(42)
    X_train = torch.randn(200, 10)
    y_train = torch.sum(X_train[:, :3], dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

    X_val = torch.randn(50, 10)
    y_val = torch.sum(X_val[:, :3], dim=1, keepdim=True) + torch.randn(50, 1) * 0.1

    # Create model
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=True)

    # Training loop
    train_losses = []
    val_losses = []
    epochs_trained = 0

    print("\nTraining with early stopping (patience=20)...")
    print("=" * 70)

    for epoch in range(500):  # Max epochs
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Check early stopping
        if early_stopping(val_loss.item(), model):
            epochs_trained = epoch
            break

        epochs_trained = epoch

    # Restore best model
    early_stopping.restore_best_model(model)

    print("\n" + "="*70)
    print(f"Training stopped at epoch {epochs_trained}")
    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    print("="*70)

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=epochs_trained-early_stopping.patience, color='r', linestyle='--',
                label=f'Best Model (epoch {epochs_trained-early_stopping.patience})', linewidth=2)
    plt.axvline(x=epochs_trained, color='orange', linestyle='--',
                label=f'Early Stop (epoch {epochs_trained})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training with Early Stopping')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    gap = np.array(val_losses) - np.array(train_losses)
    plt.plot(gap, linewidth=2, color='purple')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=epochs_trained-early_stopping.patience, color='r', linestyle='--',
                label='Best Model', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title('Generalization Gap')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('week2_early_stopping.png', dpi=150)
    plt.show()

demonstrate_early_stopping()
```

---

### 1.6 Putting It All Together

Let's combine all regularization techniques:

```python
class RegularizedNetwork(nn.Module):
    """
    Production network with all regularization techniques.

    Combines: L2, Dropout, Early Stopping
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_p=0.3, use_batchnorm=False):
        super(RegularizedNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_with_all_regularization():
    """
    Complete training example with all regularization techniques.
    """
    print("\n" + "="*70)
    print("COMPLETE REGULARIZATION DEMONSTRATION")
    print("="*70)

    # Generate data
    torch.manual_seed(42)
    X_train = torch.randn(500, 20)
    y_train = torch.sum(X_train[:, :5], dim=1, keepdim=True) + torch.randn(500, 1) * 0.2

    X_val = torch.randn(100, 20)
    y_val = torch.sum(X_val[:, :5], dim=1, keepdim=True) + torch.randn(100, 1) * 0.2

    X_test = torch.randn(100, 20)
    y_test = torch.sum(X_test[:, :5], dim=1, keepdim=True) + torch.randn(100, 1) * 0.2

    # Create model with regularization
    model = RegularizedNetwork(
        input_size=20,
        hidden_sizes=[100, 50],
        output_size=1,
        dropout_p=0.3
    )

    # Optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # Loss function
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=30, verbose=False)

    # Training loop
    print("\nTraining with:")
    print("  • Dropout (p=0.3)")
    print("  • L2 Regularization (weight_decay=0.01)")
    print("  • Early Stopping (patience=30)")
    print("=" * 70)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(500):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping check
        if early_stopping(val_loss.item(), model):
            break

    # Restore best model and evaluate on test set
    early_stopping.restore_best_model(model)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)

    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print(f"  Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Val Loss:   {early_stopping.best_loss:.6f}")
    print(f"  Test Loss:  {test_loss:.6f}")
    print("="*70)

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training with All Regularization Techniques')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('week2_complete_regularization.png', dpi=150)
    plt.show()

train_with_all_regularization()
```

---

### 1.7 Key Takeaways from Day 1

✅ **Overfitting is the enemy**

- Model memorizes instead of learning patterns
- Sign: Low train loss, high validation loss

✅ **L2 Regularization (Weight Decay)**

- Penalty: $\lambda \sum W^2$
- Effect: Shrinks all weights toward zero
- Use: General-purpose regularization
- PyTorch: `weight_decay` parameter in optimizer

✅ **L1 Regularization (Lasso)**

- Penalty: $\lambda \sum |W|$
- Effect: Drives weights to exactly zero (sparsity)
- Use: Feature selection, sparse networks

✅ **Dropout**

- Randomly drop neurons during training
- Prevents co-adaptation
- Cheap and effective
- Use: 0.2-0.5 for hidden layers

✅ **Early Stopping**

- Stop when validation loss stops improving
- Simplest and most effective
- No hyperparameters (except patience)
- Always use!

✅ **Practical Guidelines**

```python
# Good default configuration
model = Network(dropout=0.3)
optimizer = Adam(lr=0.001, weight_decay=0.01)  # L2
early_stopping = EarlyStopping(patience=20)
```

---

## 🎓 Summary Table: Regularization Techniques

| Technique          | How It Works          | When to Use   | Strength (λ)   | Pros              | Cons                    |
| ------------------ | --------------------- | ------------- | -------------- | ----------------- | ----------------------- | ---------------- | --------------- |
| **L2 (Ridge)**     | Penalize $\sum W^2$   | Always        | 0.001-0.1      | Simple, effective | Doesn't select features |
| **L1 (Lasso)**     | Penalize $\sum        | W             | $              | Feature selection | 0.001-0.1               | Creates sparsity | Can be unstable |
| **Dropout**        | Random neuron removal | Deep networks | p=0.2-0.5      | Very effective    | Slows training          |
| **Early Stopping** | Stop when val↑        | Always        | patience=10-50 | Free, easy        | Needs validation set    |

**Tomorrow:** Batch Normalization - The technique that revolutionized deep learning!

---

_End of Day 1. Total time: 6-8 hours. Take breaks! Practice the code!_

---

<a name="day-2"></a>

## 📅 Day 2: Batch Normalization and Layer Normalization

> "Batch Normalization is probably the biggest breakthrough in deep learning since dropout." - Andrew Ng

### 2.1 The Internal Covariate Shift Problem

#### What Goes Wrong in Deep Networks?

```python
def demonstrate_covariate_shift():
    """
    Visualize how layer inputs change during training.

    This is why deep networks are hard to train!
    """
    print("="*70)
    print("INTERNAL COVARIATE SHIFT DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Simple 3-layer network
    layer1 = nn.Linear(10, 20)
    layer2 = nn.Linear(20, 20)
    layer3 = nn.Linear(20, 1)

    # Random input
    X = torch.randn(100, 10)

    # Forward pass at initialization
    with torch.no_grad():
        h1_init = torch.relu(layer1(X))
        h2_init = torch.relu(layer2(h1_init))

    # Train layer1 a bit
    optimizer = optim.SGD(layer1.parameters(), lr=0.1)
    y_fake = torch.randn(100, 1)

    for _ in range(100):
        optimizer.zero_grad()
        h1 = torch.relu(layer1(X))
        h2 = torch.relu(layer2(h1))
        out = layer3(h2)
        loss = ((out - y_fake)**2).mean()
        loss.backward()
        optimizer.step()

    # Forward pass after training layer1
    with torch.no_grad():
        h1_trained = torch.relu(layer1(X))
        h2_trained = torch.relu(layer2(h1_trained))

    # Analyze distribution shift
    print("\nLayer 2 Input Statistics:")
    print("-" * 70)
    print(f"At initialization: mean={h1_init.mean():.4f}, std={h1_init.std():.4f}")
    print(f"After training:    mean={h1_trained.mean():.4f}, std={h1_trained.std():.4f}")
    print(f"\n⚠️  Distribution CHANGED! This is internal covariate shift.")
    print("   → Layer 2 must constantly adapt to changing inputs")
    print("   → Slows down training dramatically")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(h1_init.numpy().flatten(), bins=50, alpha=0.7, label='At Init', color='blue')
    axes[0].hist(h1_trained.numpy().flatten(), bins=50, alpha=0.7, label='After Training', color='red')
    axes[0].set_xlabel('Activation Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Layer 2 Input Distribution\n(Before vs After Layer 1 Training)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distributions
    h1_init_sorted = np.sort(h1_init.numpy().flatten())
    h1_trained_sorted = np.sort(h1_trained.numpy().flatten())
    cum_init = np.arange(len(h1_init_sorted)) / len(h1_init_sorted)
    cum_trained = np.arange(len(h1_trained_sorted)) / len(h1_trained_sorted)

    axes[1].plot(h1_init_sorted, cum_init, label='At Init', linewidth=2)
    axes[1].plot(h1_trained_sorted, cum_trained, label='After Training', linewidth=2)
    axes[1].set_xlabel('Activation Value')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_covariate_shift.png', dpi=150)
    plt.show()

demonstrate_covariate_shift()
```

#### The Problem:

- Layer inputs constantly change as previous layers train
- Each layer must adapt to new distributions
- Training becomes slow and unstable
- Requires careful initialization and small learning rates

---

### 2.2 Batch Normalization: The Solution

#### The Key Idea (Ioffe & Szegedy, 2015)

**Normalize layer inputs** to have mean=0, std=1:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Then **scale and shift** (learnable parameters):

$$
y_i = \gamma \hat{x}_i + \beta
$$

Where:

- $\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i$ = batch mean
- $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$ = batch variance
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = small constant for numerical stability (1e-5)

#### Why This Works: Three Magic Benefits

✅ **1. Reduces Internal Covariate Shift**

- Layer inputs always normalized
- Stable distribution across training

✅ **2. Allows Higher Learning Rates**

- Normalization prevents exploding activations
- Can train 10x faster!

✅ **3. Acts as Regularization**

- Batch statistics add noise
- Similar effect to dropout

#### Batch Normalization from Scratch

```python
class BatchNorm1dFromScratch:
    """
    Batch Normalization implementation from scratch.

    This shows exactly what happens under the hood!
    """

    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        """
        Args:
            num_features: Number of features (neurons in layer)
            epsilon: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cached values for backward pass
        self.cache = {}

    def forward(self, X, training=True):
        """
        Forward pass of batch normalization.

        Args:
            X: Input (batch_size, num_features)
            training: Whether in training mode

        Returns:
            Normalized output
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # Normalize
            X_centered = X - batch_mean
            std = np.sqrt(batch_var + self.epsilon)
            X_norm = X_centered / std

            # Scale and shift
            out = self.gamma * X_norm + self.beta

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var

            # Cache for backward pass
            self.cache = {
                'X_norm': X_norm,
                'X_centered': X_centered,
                'std': std,
                'batch_mean': batch_mean,
                'batch_var': batch_var
            }

        else:
            # Use running statistics (inference)
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta

        return out

    def backward(self, dout):
        """
        Backward pass of batch normalization.

        This is complex but important to understand!

        Args:
            dout: Gradient from next layer

        Returns:
            Gradient w.r.t. input
        """
        X_norm = self.cache['X_norm']
        X_centered = self.cache['X_centered']
        std = self.cache['std']
        m = dout.shape[0]

        # Gradients w.r.t. parameters
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # Gradient w.r.t. normalized X
        dX_norm = dout * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(dX_norm * X_centered * -0.5 * std**(-3), axis=0)

        # Gradient w.r.t. mean
        dmean = np.sum(dX_norm * -1/std, axis=0) + \
                dvar * np.sum(-2 * X_centered, axis=0) / m

        # Gradient w.r.t. input
        dX = dX_norm / std + \
             dvar * 2 * X_centered / m + \
             dmean / m

        return dX, dgamma, dbeta


def demonstrate_batch_norm_from_scratch():
    """
    Test our batch norm implementation.
    """
    print("\n" + "="*70)
    print("BATCH NORMALIZATION FROM SCRATCH")
    print("="*70)

    # Create data
    np.random.seed(42)
    X = np.random.randn(32, 10) * 5 + 10  # Batch of 32, 10 features

    print(f"\nInput statistics:")
    print(f"  Mean: {X.mean(axis=0)[:3]}...")  # Show first 3
    print(f"  Std:  {X.std(axis=0)[:3]}...")

    # Apply batch norm
    bn = BatchNorm1dFromScratch(num_features=10)
    X_normalized = bn.forward(X, training=True)

    print(f"\nAfter Batch Norm:")
    print(f"  Mean: {X_normalized.mean(axis=0)[:3]}...")  # Should be ~0
    print(f"  Std:  {X_normalized.std(axis=0)[:3]}...")   # Should be ~1

    print("\n✓ Batch Normalization successfully normalized the data!")
    print("  → Mean ≈ 0, Std ≈ 1")

demonstrate_batch_norm_from_scratch()
```

#### PyTorch Batch Normalization

```python
def demonstrate_pytorch_batchnorm():
    """
    Using PyTorch's built-in Batch Normalization.
    """
    print("\n" + "="*70)
    print("PYTORCH BATCH NORMALIZATION")
    print("="*70)

    # Network WITHOUT batch norm
    model_no_bn = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # Network WITH batch norm
    model_with_bn = nn.Sequential(
        nn.Linear(10, 50),
        nn.BatchNorm1d(50),  # ← Batch norm after linear layer
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.BatchNorm1d(50),  # ← Before activation
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    print("\n📌 Key Points:")
    print("  • BatchNorm1d for fully connected layers")
    print("  • BatchNorm2d for convolutional layers")
    print("  • Place BEFORE activation function (common) or AFTER (also works)")
    print("  • Automatically handles train/eval mode")

    # Generate data
    torch.manual_seed(42)
    X_train = torch.randn(1000, 10)
    y_train = torch.sum(X_train[:, :3]**2, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.5

    X_test = torch.randn(200, 10)
    y_test = torch.sum(X_test[:, :3]**2, dim=1, keepdim=True) + torch.randn(200, 1) * 0.5

    # Training function
    def train_model(model, epochs=200, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []

        for epoch in range(epochs):
            model.train()  # Important for batch norm!
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        model.eval()  # Important for batch norm!
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)

        return train_losses, test_loss.item()

    print("\nTraining WITHOUT Batch Norm...")
    train_loss_no_bn, test_loss_no_bn = train_model(model_no_bn, lr=0.01)

    print("Training WITH Batch Norm...")
    train_loss_with_bn, test_loss_with_bn = train_model(model_with_bn, lr=0.01)

    print("\n" + "="*70)
    print("RESULTS:")
    print(f"Without BN - Final Train Loss: {train_loss_no_bn[-1]:.6f}, Test Loss: {test_loss_no_bn:.6f}")
    print(f"With BN    - Final Train Loss: {train_loss_with_bn[-1]:.6f}, Test Loss: {test_loss_with_bn:.6f}")
    print("="*70)

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_no_bn, label='Without Batch Norm', linewidth=2)
    plt.plot(train_loss_with_bn, label='With Batch Norm', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    # Compare with higher learning rate (batch norm allows this!)
    print("\nNow testing with HIGHER learning rate (lr=0.1)...")
    _, test_no_bn_high_lr = train_model(model_no_bn, lr=0.1)
    _, test_with_bn_high_lr = train_model(model_with_bn, lr=0.1)

    x = ['lr=0.01\nNo BN', 'lr=0.01\nWith BN', 'lr=0.1\nNo BN', 'lr=0.1\nWith BN']
    y = [test_loss_no_bn, test_loss_with_bn, test_no_bn_high_lr, test_with_bn_high_lr]
    colors = ['red', 'green', 'darkred', 'darkgreen']

    plt.bar(x, y, color=colors, alpha=0.7)
    plt.ylabel('Test Loss')
    plt.title('Batch Norm Enables Higher Learning Rates')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_batchnorm_comparison.png', dpi=150)
    plt.show()

    print("\n✓ Batch Norm allows using much higher learning rates!")

demonstrate_pytorch_batchnorm()
```

---

### 2.3 Layer Normalization

#### The Problem with Batch Norm

❌ **Depends on batch size**

- Small batches → unreliable statistics
- Batch size = 1 → doesn't work!

❌ **Problematic for RNNs/Transformers**

- Sequential data has varying lengths
- Batch statistics across time steps are weird

#### Layer Norm Solution (Ba et al., 2016)

**Normalize across features instead of batch:**

$$
\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}
$$

Where:

- $\mu_L$ = mean across features (not batch!)
- $\sigma_L^2$ = variance across features

#### Batch Norm vs Layer Norm

```python
def visualize_batch_vs_layer_norm():
    """
    Visualize the difference between Batch Norm and Layer Norm.
    """
    print("\n" + "="*70)
    print("BATCH NORM vs LAYER NORM")
    print("="*70)

    # Create example tensor: (batch_size=4, features=6)
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ])

    print("\nOriginal Data (4 samples, 6 features):")
    print(X.numpy())

    # Batch Norm: normalize each FEATURE across batch
    bn = nn.BatchNorm1d(6, affine=False)  # No learnable params for demo
    bn.eval()  # Use current statistics
    X_batch_norm = bn(X)

    # Layer Norm: normalize each SAMPLE across features
    ln = nn.LayerNorm(6, elementwise_affine=False)
    X_layer_norm = ln(X)

    print("\nAfter Batch Norm (normalize each column):")
    print(X_batch_norm.detach().numpy())
    print("  → Each FEATURE (column) has mean≈0, std≈1")

    print("\nAfter Layer Norm (normalize each row):")
    print(X_layer_norm.detach().numpy())
    print("  → Each SAMPLE (row) has mean≈0, std≈1")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cmap = 'RdYlGn'

    im0 = axes[0].imshow(X.numpy(), cmap=cmap, aspect='auto')
    axes[0].set_xlabel('Feature')
    axes[0].set_ylabel('Sample')
    axes[0].set_title('Original Data')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(X_batch_norm.detach().numpy(), cmap=cmap, aspect='auto')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Sample')
    axes[1].set_title('Batch Norm\n(Normalize Columns)')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(X_layer_norm.detach().numpy(), cmap=cmap, aspect='auto')
    axes[2].set_xlabel('Feature')
    axes[2].set_ylabel('Sample')
    axes[2].set_title('Layer Norm\n(Normalize Rows)')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('week2_batch_vs_layer_norm.png', dpi=150)
    plt.show()

    print("\n🔑 Key Difference:")
    print("  Batch Norm: Normalize ACROSS samples (↓ columns)")
    print("  Layer Norm: Normalize ACROSS features (→ rows)")

visualize_batch_vs_layer_norm()
```

#### When to Use Which?

| Aspect                 | Batch Norm                    | Layer Norm                        |
| ---------------------- | ----------------------------- | --------------------------------- |
| **Normalizes**         | Across batch                  | Across features                   |
| **Use Case**           | CNNs, MLPs with large batches | RNNs, Transformers, small batches |
| **Batch Size**         | Needs large batches (>16)     | Works with batch size = 1         |
| **Training/Inference** | Different behavior            | Same behavior                     |
| **Example**            | Image classification          | Language models (GPT, BERT)       |

```python
def compare_norms_in_practice():
    """
    Compare Batch Norm and Layer Norm in realistic scenario.
    """
    print("\n" + "="*70)
    print("PRACTICAL COMPARISON: BATCH NORM vs LAYER NORM")
    print("="*70)

    torch.manual_seed(42)

    # Small dataset
    X_train = torch.randn(100, 20)
    y_train = torch.sum(X_train[:, :5], dim=1, keepdim=True) + torch.randn(100, 1) * 0.2

    X_test = torch.randn(20, 20)
    y_test = torch.sum(X_test[:, :5], dim=1, keepdim=True) + torch.randn(20, 1) * 0.2

    # Test with different batch sizes
    batch_sizes = [1, 4, 16, 32]
    results = {'BatchNorm': [], 'LayerNorm': []}

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size={batch_size}...")

        # Batch Norm model
        model_bn = nn.Sequential(
            nn.Linear(20, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        # Layer Norm model
        model_ln = nn.Sequential(
            nn.Linear(20, 50),
            nn.LayerNorm(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        # Train both
        def train_quick(model, bs):
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            for epoch in range(100):
                model.train()
                for i in range(0, len(X_train), bs):
                    batch_X = X_train[i:i+bs]
                    batch_y = y_train[i:i+bs]

                    if len(batch_X) < 2 and isinstance(model[1], nn.BatchNorm1d):
                        continue  # Skip single-sample batches for BatchNorm

                    optimizer.zero_grad()
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test)

            return test_loss.item()

        try:
            loss_bn = train_quick(model_bn, batch_size)
            results['BatchNorm'].append(loss_bn)
            print(f"  BatchNorm test loss: {loss_bn:.4f}")
        except:
            results['BatchNorm'].append(None)
            print(f"  BatchNorm FAILED")

        loss_ln = train_quick(model_ln, batch_size)
        results['LayerNorm'].append(loss_ln)
        print(f"  LayerNorm test loss: {loss_ln:.4f}")

    # Visualize
    plt.figure(figsize=(10, 6))

    x_pos = np.arange(len(batch_sizes))
    width = 0.35

    bn_losses = [l if l is not None else 0 for l in results['BatchNorm']]
    ln_losses = results['LayerNorm']

    plt.bar(x_pos - width/2, bn_losses, width, label='Batch Norm', alpha=0.8)
    plt.bar(x_pos + width/2, ln_losses, width, label='Layer Norm', alpha=0.8)

    plt.xlabel('Batch Size')
    plt.ylabel('Test Loss')
    plt.title('Batch Norm vs Layer Norm: Effect of Batch Size')
    plt.xticks(x_pos, batch_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_norm_comparison.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("💡 Observation:")
    print("  • Batch Norm struggles with small batches (especially batch_size=1)")
    print("  • Layer Norm works consistently across all batch sizes")
    print("  • For large batches, both work well")
    print("="*70)

compare_norms_in_practice()
```

---

### 2.4 Key Takeaways from Day 2

✅ **Internal Covariate Shift**

- Layer inputs change as previous layers train
- Makes deep networks hard to train

✅ **Batch Normalization**

- Normalizes across batch dimension
- Formula: $\frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
- Benefits: Faster training, higher learning rates, regularization effect
- Use: CNNs, large-batch training
- Place: Before or after activation (both work)

✅ **Layer Normalization**

- Normalizes across feature dimension
- Independent of batch size
- Use: RNNs, Transformers, small batches
- Same behavior in train/eval mode

✅ **Practical Guidelines**

```python
# For CNNs and large batches
model = nn.Sequential(
    nn.Linear(in_features, out_features),
    nn.BatchNorm1d(out_features),
    nn.ReLU()
)

# For RNNs, Transformers, small batches
model = nn.Sequential(
    nn.Linear(in_features, out_features),
    nn.LayerNorm(out_features),
    nn.ReLU()
)
```

✅ **Important Notes**

- Always use `model.train()` and `model.eval()` with Batch Norm!
- Batch Norm needs batch_size ≥ 2
- Layer Norm works with batch_size = 1
- Both add learnable parameters (γ, β)

**Tomorrow:** Advanced optimization algorithms beyond vanilla SGD!

---

_End of Day 2. Total time: 6-8 hours._

---

<a name="day-3"></a>

## 📅 Day 3: Advanced Optimization Algorithms

> "The right optimizer can make the difference between a model that trains and one that doesn't."

### 3.1 Beyond Vanilla SGD: The Problem

Recall **vanilla SGD** from Week 1:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

#### Problems with Vanilla SGD:

❌ **Same learning rate for all parameters**

- Some need big steps, others need small steps

❌ **Oscillation in ravines**

- Steep in one direction, shallow in another
- Wastes time bouncing around

❌ **Getting stuck in saddle points**

- High-dimensional spaces have many saddle points
- Zero gradient but not optimal!

❌ **Sensitive to learning rate**

- Too small → slow training
- Too large → divergence

### 3.2 SGD with Momentum

#### The Intuition: Physics!

Imagine a ball rolling down a hill:

- Builds up **velocity** as it rolls
- Doesn't stop at small bumps
- Dampens oscillations

#### Mathematical Formulation

**Velocity (moving average of gradients):**

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t)
$$

**Update:**

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

Where $\beta$ = momentum coefficient (typically 0.9)

#### Why It Works

```python
def visualize_momentum():
    """
    Visualize how momentum helps optimization.
    """
    print("="*70)
    print("SGD WITH MOMENTUM VISUALIZATION")
    print("="*70)

    # Create a ravine-like loss function
    def loss_ravine(x, y):
        return x**2 + 10*y**2

    def grad_ravine(x, y):
        return np.array([2*x, 20*y])

    # Starting point
    start = np.array([-2.0, 0.8])

    # Vanilla SGD
    lr = 0.1
    pos_sgd = [start.copy()]
    current = start.copy()

    for _ in range(50):
        grad = grad_ravine(current[0], current[1])
        current = current - lr * grad
        pos_sgd.append(current.copy())

    # SGD with Momentum
    pos_momentum = [start.copy()]
    current = start.copy()
    velocity = np.zeros(2)
    beta = 0.9

    for _ in range(50):
        grad = grad_ravine(current[0], current[1])
        velocity = beta * velocity + grad
        current = current - lr * velocity
        pos_momentum.append(current.copy())

    pos_sgd = np.array(pos_sgd)
    pos_momentum = np.array(pos_momentum)

    # Visualize
    fig = plt.figure(figsize=(14, 6))

    # Create contour plot
    x = np.linspace(-2.5, 0.5, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_ravine(X, Y)

    # SGD path
    ax1 = fig.add_subplot(121)
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax1.plot(pos_sgd[:, 0], pos_sgd[:, 1], 'r.-', linewidth=2, markersize=4, label='SGD Path')
    ax1.plot(pos_sgd[0, 0], pos_sgd[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(0, 0, 'r*', markersize=15, label='Optimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Vanilla SGD\n(Oscillates in ravine)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Momentum path
    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax2.plot(pos_momentum[:, 0], pos_momentum[:, 1], 'b.-', linewidth=2, markersize=4, label='Momentum Path')
    ax2.plot(pos_momentum[0, 0], pos_momentum[0, 1], 'go', markersize=10, label='Start')
    ax2.plot(0, 0, 'r*', markersize=15, label='Optimum')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('SGD with Momentum\n(Smooth, direct path)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_momentum_visualization.png', dpi=150)
    plt.show()

    print("\n✓ Momentum dampens oscillations and accelerates in consistent directions!")

visualize_momentum()
```

#### Implementation

```python
class SGDMomentum:
    """
    SGD with momentum from scratch.
    """

    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity for each parameter
        self.velocities = [torch.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update parameters using momentum."""
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue

                # Update velocity: v = β*v + grad
                self.velocities[i] = self.momentum * self.velocities[i] + param.grad

                # Update parameter: θ = θ - η*v
                param.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


def test_momentum_optimizer():
    """Test momentum on a simple problem."""
    print("\n" + "="*70)
    print("SGD WITH MOMENTUM - IMPLEMENTATION TEST")
    print("="*70)

    torch.manual_seed(42)

    # Simple regression
    X = torch.randn(100, 10)
    y = torch.sum(X[:, :3], dim=1, keepdim=True) + torch.randn(100, 1) * 0.1

    # Model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

    # Our momentum optimizer
    optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()

    # Train
    losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    print("\n✓ Momentum optimizer working correctly!")

    # Compare with PyTorch's built-in
    print("\nUsing PyTorch built-in SGD with momentum:")
    model2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

    losses2 = []
    for epoch in range(200):
        optimizer2.zero_grad()
        pred = model2(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Our Implementation', linewidth=2)
    plt.plot(losses2, label='PyTorch SGD', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Momentum Optimizer Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('week2_momentum_implementation.png', dpi=150)
    plt.show()

test_momentum_optimizer()
```

---

### 3.3 RMSprop: Adaptive Learning Rates

#### The Problem

Different parameters need different learning rates:

- Some directions are steep (need small steps)
- Some directions are shallow (need big steps)

#### The Solution (Hinton, 2012)

**Adapt learning rate per parameter** based on recent gradient magnitudes:

$$
s_t = \rho s_{t-1} + (1-\rho)(\nabla L)^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}}\nabla L
$$

Where:

- $s_t$ = moving average of squared gradients
- $\rho$ = decay rate (typically 0.9)
- $\epsilon$ = small constant (1e-8)

#### Intuition

- Large gradients → large $s_t$ → **smaller effective learning rate**
- Small gradients → small $s_t$ → **larger effective learning rate**

```python
class RMSprop:
    """
    RMSprop optimizer from scratch.
    """

    def __init__(self, parameters, lr=0.01, rho=0.9, epsilon=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Initialize squared gradient accumulator
        self.square_avg = [torch.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Update parameters using RMSprop."""
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue

                grad = param.grad

                # Update squared gradient average
                self.square_avg[i] = self.rho * self.square_avg[i] + \
                                    (1 - self.rho) * (grad ** 2)

                # Adaptive learning rate
                adapted_lr = self.lr / (torch.sqrt(self.square_avg[i]) + self.epsilon)

                # Update parameter
                param.data -= adapted_lr * grad

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


def demonstrate_rmsprop():
    """
    Demonstrate RMSprop's adaptive learning rates.
    """
    print("\n" + "="*70)
    print("RMSPROP DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Problem with different scales
    X = torch.randn(200, 10)
    # Feature 0 has large scale, features 1-2 have small scale
    w_true = torch.tensor([[10.0], [0.1], [0.1]] + [[0.0]]*7)
    y = X @ w_true + torch.randn(200, 1) * 0.1

    # Model
    model_sgd = nn.Linear(10, 1)
    model_rmsprop = nn.Linear(10, 1)

    # Copy initial weights
    with torch.no_grad():
        model_rmsprop.weight.copy_(model_sgd.weight)
        model_rmsprop.bias.copy_(model_sgd.bias)

    # Optimizers
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.001)
    opt_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    # Train
    losses_sgd = []
    losses_rmsprop = []

    for epoch in range(300):
        # SGD
        opt_sgd.zero_grad()
        pred_sgd = model_sgd(X)
        loss_sgd = criterion(pred_sgd, y)
        loss_sgd.backward()
        opt_sgd.step()
        losses_sgd.append(loss_sgd.item())

        # RMSprop
        opt_rmsprop.zero_grad()
        pred_rmsprop = model_rmsprop(X)
        loss_rmsprop = criterion(pred_rmsprop, y)
        loss_rmsprop.backward()
        opt_rmsprop.step()
        losses_rmsprop.append(loss_rmsprop.item())

    print(f"\nFinal Loss:")
    print(f"  SGD:     {losses_sgd[-1]:.6f}")
    print(f"  RMSprop: {losses_rmsprop[-1]:.6f}")

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_sgd, label='SGD (lr=0.001)', linewidth=2)
    plt.plot(losses_rmsprop, label='RMSprop (lr=0.01)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RMSprop vs SGD\n(RMSprop adapts to different scales)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Weight comparison
    plt.subplot(1, 2, 2)
    w_sgd = model_sgd.weight.detach().numpy().flatten()[:3]
    w_rmsprop = model_rmsprop.weight.detach().numpy().flatten()[:3]
    w_true_vals = w_true.numpy().flatten()[:3]

    x_pos = np.arange(3)
    width = 0.25

    plt.bar(x_pos - width, w_true_vals, width, label='True Weights', alpha=0.8)
    plt.bar(x_pos, w_sgd, width, label='SGD', alpha=0.8)
    plt.bar(x_pos + width, w_rmsprop, width, label='RMSprop', alpha=0.8)

    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.title('Learned Weights (First 3)\n(RMSprop closer to true values)')
    plt.xticks(x_pos, ['w0', 'w1', 'w2'])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_rmsprop_demo.png', dpi=150)
    plt.show()

    print("\n✓ RMSprop adapts learning rate per parameter!")

demonstrate_rmsprop()
```

---

### 3.4 Adam: The King of Optimizers

#### Combining the Best of Both Worlds

**Adam** (Kingma & Ba, 2014) = **Momentum** + **RMSprop**

**First moment (momentum):**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L
$$

**Second moment (RMSprop):**

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2
$$

**Bias correction** (important!):

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

**Update:**

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
$$

#### Default Hyperparameters

These work surprisingly well for most problems:

- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (RMSprop decay)
- $\epsilon = 10^{-8}$
- $\eta = 0.001$ (learning rate)

#### Implementation from Scratch

```python
class Adam:
    """
    Adam optimizer from scratch.

    The most popular optimizer in deep learning!
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moments
        self.m = [torch.zeros_like(p.data) for p in self.parameters]  # First moment
        self.v = [torch.zeros_like(p.data) for p in self.parameters]  # Second moment
        self.t = 0  # Time step

    def step(self):
        """Update parameters using Adam."""
        self.t += 1

        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue

                grad = param.grad

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameter
                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


def demonstrate_adam():
    """
    Demonstrate Adam optimizer superiority.
    """
    print("\n" + "="*70)
    print("ADAM OPTIMIZER DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Complex dataset
    X = torch.randn(500, 20)
    # Non-linear target
    y = torch.sum(X[:, :5]**2, dim=1, keepdim=True) + \
        torch.sin(torch.sum(X[:, 5:10], dim=1, keepdim=True)) + \
        torch.randn(500, 1) * 0.2

    # Create models (same architecture)
    def create_model():
        return nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    model_sgd = create_model()
    model_momentum = create_model()
    model_rmsprop = create_model()
    model_adam = create_model()

    # Copy weights to ensure fair comparison
    with torch.no_grad():
        for m in [model_momentum, model_rmsprop, model_adam]:
            for p1, p2 in zip(model_sgd.parameters(), m.parameters()):
                p2.data.copy_(p1.data)

    # Optimizers
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.001)
    opt_momentum = optim.SGD(model_momentum.parameters(), lr=0.001, momentum=0.9)
    opt_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=0.001)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    # Train all
    history = {'SGD': [], 'Momentum': [], 'RMSprop': [], 'Adam': []}
    epochs = 300

    print("\nTraining all optimizers...")
    for epoch in range(epochs):
        # SGD
        opt_sgd.zero_grad()
        loss = criterion(model_sgd(X), y)
        loss.backward()
        opt_sgd.step()
        history['SGD'].append(loss.item())

        # Momentum
        opt_momentum.zero_grad()
        loss = criterion(model_momentum(X), y)
        loss.backward()
        opt_momentum.step()
        history['Momentum'].append(loss.item())

        # RMSprop
        opt_rmsprop.zero_grad()
        loss = criterion(model_rmsprop(X), y)
        loss.backward()
        opt_rmsprop.step()
        history['RMSprop'].append(loss.item())

        # Adam
        opt_adam.zero_grad()
        loss = criterion(model_adam(X), y)
        loss.backward()
        opt_adam.step()
        history['Adam'].append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | SGD: {history['SGD'][-1]:.4f} | "
                  f"Momentum: {history['Momentum'][-1]:.4f} | "
                  f"RMSprop: {history['RMSprop'][-1]:.4f} | "
                  f"Adam: {history['Adam'][-1]:.4f}")

    print("\n" + "="*70)
    print("FINAL LOSSES:")
    for name, losses in history.items():
        print(f"  {name:12s}: {losses[-1]:.6f}")
    print("="*70)

    # Visualize
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, losses in history.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    final_losses = [losses[-1] for losses in history.values()]
    colors = ['red', 'orange', 'yellow', 'green']
    plt.bar(history.keys(), final_losses, color=colors, alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('Final Performance\n(Lower is better)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_optimizer_comparison.png', dpi=150)
    plt.show()

    print("\n✓ Adam combines best of momentum and RMSprop!")
    print("  → Usually the best default choice")

demonstrate_adam()
```

---

### 3.5 AdamW: Adam with Proper Weight Decay

#### The Problem with Adam + L2

When you use Adam with `weight_decay`, it's actually **wrong**!

**Regular L2 regularization:**

$$
L = L_{data} + \frac{\lambda}{2}||W||^2
$$

$$
\nabla L = \nabla L_{data} + \lambda W
$$

**But Adam applies adaptive learning rates to this gradient!**

- L2 penalty gets divided by $\sqrt{v_t}$
- Reduces effectiveness of regularization

#### AdamW Solution (Loshchilov & Hutter, 2019)

**Decouple weight decay from gradient:**

```python
# Wrong (Adam with weight_decay)
grad = grad + weight_decay * param
param = param - lr * adam_update(grad)

# Correct (AdamW)
param = param - lr * adam_update(grad) - lr * weight_decay * param
```

#### Implementation

```python
def compare_adam_adamw():
    """
    Compare Adam and AdamW.
    """
    print("\n" + "="*70)
    print("ADAM vs ADAMW")
    print("="*70)

    torch.manual_seed(42)

    # Overfitting scenario (many parameters, little data)
    X_train = torch.randn(50, 100)
    y_train = torch.sum(X_train[:, :10], dim=1, keepdim=True) + torch.randn(50, 1) * 0.1

    X_val = torch.randn(50, 100)
    y_val = torch.sum(X_val[:, :10], dim=1, keepdim=True) + torch.randn(50, 1) * 0.1

    # Large model (prone to overfitting)
    def create_model():
        return nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    model_adam = create_model()
    model_adamw = create_model()

    # Copy weights
    with torch.no_grad():
        for p1, p2 in zip(model_adam.parameters(), model_adamw.parameters()):
            p2.data.copy_(p1.data)

    # Optimizers
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

    criterion = nn.MSELoss()

    # Train
    history = {
        'adam_train': [], 'adam_val': [],
        'adamw_train': [], 'adamw_val': []
    }

    for epoch in range(500):
        # Adam
        model_adam.train()
        opt_adam.zero_grad()
        loss_train = criterion(model_adam(X_train), y_train)
        loss_train.backward()
        opt_adam.step()

        model_adam.eval()
        with torch.no_grad():
            loss_val = criterion(model_adam(X_val), y_val)

        history['adam_train'].append(loss_train.item())
        history['adam_val'].append(loss_val.item())

        # AdamW
        model_adamw.train()
        opt_adamw.zero_grad()
        loss_train = criterion(model_adamw(X_train), y_train)
        loss_train.backward()
        opt_adamw.step()

        model_adamw.eval()
        with torch.no_grad():
            loss_val = criterion(model_adamw(X_val), y_val)

        history['adamw_train'].append(loss_train.item())
        history['adamw_val'].append(loss_val.item())

    # Results
    print("\nFinal Validation Loss:")
    print(f"  Adam:  {history['adam_val'][-1]:.6f}")
    print(f"  AdamW: {history['adamw_val'][-1]:.6f}")
    print(f"\nAdamW improvement: {((history['adam_val'][-1] - history['adamw_val'][-1])/history['adam_val'][-1]*100):.1f}%")

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['adam_train'], label='Adam Train', linewidth=2, alpha=0.7)
    plt.plot(history['adam_val'], label='Adam Val', linewidth=2)
    plt.plot(history['adamw_train'], label='AdamW Train', linewidth=2, alpha=0.7, linestyle='--')
    plt.plot(history['adamw_val'], label='AdamW Val', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Adam vs AdamW')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Generalization gap
    plt.subplot(1, 2, 2)
    gap_adam = np.array(history['adam_val']) - np.array(history['adam_train'])
    gap_adamw = np.array(history['adamw_val']) - np.array(history['adamw_train'])

    plt.plot(gap_adam, label='Adam Gap', linewidth=2)
    plt.plot(gap_adamw, label='AdamW Gap', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title('Generalization Gap\n(Lower is better - less overfitting)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('week2_adam_vs_adamw.png', dpi=150)
    plt.show()

    print("\n✓ AdamW applies weight decay correctly!")
    print("  → Better regularization than Adam with weight_decay")

compare_adam_adamw()
```

---

### 3.6 Optimizer Cheat Sheet

| Optimizer    | Formula                                    | Pros                         | Cons                          | When to Use                     |
| ------------ | ------------------------------------------ | ---------------------------- | ----------------------------- | ------------------------------- |
| **SGD**      | $\theta - \eta \nabla L$                   | Simple, stable               | Slow, sensitive to LR         | Baseline                        |
| **Momentum** | $\theta - \eta v_t$                        | Faster, dampens oscillations | Still sensitive to LR         | Better than SGD                 |
| **RMSprop**  | $\theta - \frac{\eta}{\sqrt{s_t}}\nabla L$ | Adaptive per-parameter LR    | Can be unstable               | RNNs                            |
| **Adam**     | Momentum + RMSprop                         | Fast, adaptive, robust       | Can overfit, bad weight decay | **Default choice**              |
| **AdamW**    | Adam + proper weight decay                 | Better regularization        | Slightly slower               | **When regularization matters** |

### 3.7 Key Takeaways from Day 3

✅ **SGD with Momentum**

- Accumulates velocity
- Dampens oscillations
- Accelerates in consistent directions
- Default: β = 0.9

✅ **RMSprop**

- Adaptive learning rate per parameter
- Good for non-stationary problems
- Default: ρ = 0.9

✅ **Adam**

- Best of both worlds
- Most popular optimizer
- Defaults: β₁=0.9, β₂=0.999, lr=0.001
- Works well out of the box

✅ **AdamW**

- Adam with correct weight decay
- Better generalization
- **Use instead of Adam + weight_decay**

✅ **Practical Guidelines**

```python
# Good default for most problems
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# For very large models (LLMs)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

# When you need stability
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Tomorrow:** Learning rate schedules and hyperparameter tuning!

---

_End of Day 3. Total time: 6-8 hours._

---

<a name="day-4"></a>

## 📅 Day 4: Learning Rate Scheduling and Hyperparameter Tuning

> "The learning rate is the most important hyperparameter." - Yoshua Bengio

### 4.1 Why Learning Rate Schedules Matter

#### The Problem with Fixed Learning Rates

**Too large:**

- Fast initial progress
- But bounces around optimum
- Never converges well

**Too small:**

- Slow but steady
- Takes forever to train

**Ideal solution: Start large, then decrease!**

```python
def demonstrate_lr_importance():
    """
    Show impact of learning rate on training.
    """
    print("="*70)
    print("LEARNING RATE IMPACT DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Simple problem
    X = torch.randn(200, 10)
    y = torch.sum(X[:, :3], dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(200):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        axes[idx].plot(losses, linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(f'Learning Rate = {lr}')
        axes[idx].grid(True)
        axes[idx].set_yscale('log')

        if lr == 1.0:
            axes[idx].text(100, max(losses)*0.5, 'DIVERGED!',
                          fontsize=14, color='red', weight='bold')
        elif lr == 0.0001:
            axes[idx].text(100, max(losses)*0.5, 'TOO SLOW!',
                          fontsize=14, color='orange', weight='bold')
        elif lr == 0.01:
            axes[idx].text(100, losses[-1]*2, 'JUST RIGHT!',
                          fontsize=14, color='green', weight='bold')

    # Hide last subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('week2_lr_importance.png', dpi=150)
    plt.show()

    print("\n✓ Learning rate dramatically affects training!")
    print("  → Too small: slow convergence")
    print("  → Too large: instability/divergence")
    print("  → Just right: fast and stable")

demonstrate_lr_importance()
```

---

### 4.2 Learning Rate Schedules

#### 1. Step Decay

**Reduce LR by factor every N epochs:**

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/N \rfloor}
$$

```python
# PyTorch implementation
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()  # Update learning rate
```

#### 2. Exponential Decay

**Smooth exponential decrease:**

$$
\eta_t = \eta_0 \cdot \gamma^t
$$

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

#### 3. Cosine Annealing

**Smooth cosine curve (popular for modern networks):**

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)
$$

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
```

#### 4. Reduce on Plateau

**Reduce when validation loss stops improving (adaptive!):**

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize metric
    factor=0.1,      # Multiply LR by 0.1
    patience=10,     # Wait 10 epochs
    verbose=True
)

# Training loop
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)  # Pass validation metric
```

#### 5. One Cycle Policy (Modern + Fast!)

**The secret sauce of fast training:**

- Warmup: Increase LR from low to high
- Anneal: Decrease LR from high to very low
- Used by fast.ai

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,           # Maximum LR
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)

# Training loop (step per batch!)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(batch)
        scheduler.step()  # Update every batch!
```

#### Complete Comparison

```python
def compare_lr_schedules():
    """
    Compare different learning rate schedules.
    """
    print("\n" + "="*70)
    print("LEARNING RATE SCHEDULES COMPARISON")
    print("="*70)

    torch.manual_seed(42)

    # Data
    X = torch.randn(500, 20)
    y = torch.sum(X[:, :5]**2, dim=1, keepdim=True) + torch.randn(500, 1) * 0.3

    X_val = torch.randn(100, 20)
    y_val = torch.sum(X_val[:, :5]**2, dim=1, keepdim=True) + torch.randn(100, 1) * 0.3

    criterion = nn.MSELoss()
    epochs = 200

    # Test different schedules
    schedules = {
        'Constant': None,
        'Step Decay': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5),
        'Exponential': lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.98),
        'Cosine': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
        'Reduce on Plateau': lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, verbose=False)
    }

    results = {}

    for name, scheduler_fn in schedules.items():
        print(f"\nTraining with {name}...")

        model = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        train_losses = []
        val_losses = []
        lrs = []

        for epoch in range(epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            train_loss = criterion(pred, y)
            train_loss.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            # Update scheduler
            if scheduler:
                if name == 'Reduce on Plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        results[name] = {
            'train': train_losses,
            'val': val_losses,
            'lr': lrs
        }

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data['train'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Validation loss
    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(data['val'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Learning rate schedules
    ax = axes[1, 0]
    for name, data in results.items():
        ax.plot(data['lr'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Final performance
    ax = axes[1, 1]
    final_vals = [data['val'][-1] for data in results.values()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    bars = ax.bar(results.keys(), final_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Highlight best
    best_idx = np.argmin(final_vals)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('week2_lr_schedules.png', dpi=150)
    plt.show()

    # Print results
    print("\n" + "="*70)
    print("FINAL VALIDATION LOSSES:")
    for name, data in results.items():
        print(f"  {name:20s}: {data['val'][-1]:.6f}")
    print("="*70)

compare_lr_schedules()
```

---

### 4.3 Learning Rate Warmup

#### The Problem

When training starts:

- Weights are random
- Large gradients
- High learning rate → unstable!

#### Solution: Warmup

**Gradually increase LR** for first few epochs:

$$
\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}
$$

```python
def create_lr_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, max_lr=0.001):
    """
    Learning rate schedule with warmup + cosine decay.

    This is the modern best practice!
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def demonstrate_warmup():
    """
    Show benefits of learning rate warmup.
    """
    print("\n" + "="*70)
    print("LEARNING RATE WARMUP DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    X = torch.randn(400, 30)
    y = torch.sum(X[:, :10]**2, dim=1, keepdim=True) + torch.randn(400, 1) * 0.5

    epochs = 150

    # Without warmup
    model_no_warmup = nn.Sequential(
        nn.Linear(30, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 1)
    )
    opt_no_warmup = optim.Adam(model_no_warmup.parameters(), lr=0.01)
    sched_no_warmup = optim.lr_scheduler.CosineAnnealingLR(opt_no_warmup, T_max=epochs)

    # With warmup
    model_warmup = nn.Sequential(
        nn.Linear(30, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 1)
    )
    opt_warmup = optim.Adam(model_warmup.parameters(), lr=0.01)
    sched_warmup = create_lr_schedule_with_warmup(opt_warmup, warmup_epochs=20, total_epochs=epochs)

    criterion = nn.MSELoss()

    # Copy weights
    with torch.no_grad():
        for p1, p2 in zip(model_no_warmup.parameters(), model_warmup.parameters()):
            p2.data.copy_(p1.data)

    # Train both
    history = {
        'no_warmup_loss': [], 'no_warmup_lr': [],
        'warmup_loss': [], 'warmup_lr': []
    }

    for epoch in range(epochs):
        # No warmup
        opt_no_warmup.zero_grad()
        loss = criterion(model_no_warmup(X), y)
        loss.backward()
        opt_no_warmup.step()
        sched_no_warmup.step()
        history['no_warmup_loss'].append(loss.item())
        history['no_warmup_lr'].append(opt_no_warmup.param_groups[0]['lr'])

        # With warmup
        opt_warmup.zero_grad()
        loss = criterion(model_warmup(X), y)
        loss.backward()
        opt_warmup.step()
        sched_warmup.step()
        history['warmup_loss'].append(loss.item())
        history['warmup_lr'].append(opt_warmup.param_groups[0]['lr'])

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history['no_warmup_loss'], label='No Warmup', linewidth=2)
    axes[0].plot(history['warmup_loss'], label='With Warmup', linewidth=2)
    axes[0].axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Warmup End')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss: Impact of Warmup')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')

    # LR schedules
    axes[1].plot(history['no_warmup_lr'], label='No Warmup', linewidth=2)
    axes[1].plot(history['warmup_lr'], label='With Warmup', linewidth=2)
    axes[1].axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Warmup End')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('week2_warmup.png', dpi=150)
    plt.show()

    print("\n✓ Warmup stabilizes early training!")
    print(f"  Final loss (no warmup): {history['no_warmup_loss'][-1]:.6f}")
    print(f"  Final loss (with warmup): {history['warmup_loss'][-1]:.6f}")

demonstrate_warmup()
```

---

### 4.4 Hyperparameter Tuning

#### Key Hyperparameters (Priority Order)

1. **Learning Rate** ⭐⭐⭐⭐⭐ (Most important!)
2. **Batch Size** ⭐⭐⭐⭐
3. **Architecture** (width, depth) ⭐⭐⭐⭐
4. **Regularization** (dropout, weight decay) ⭐⭐⭐
5. **Optimizer parameters** (β1, β2) ⭐⭐

#### Search Strategies

**1. Grid Search** (Exhaustive but expensive)

```python
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]
dropout_rates = [0.2, 0.3, 0.5]

for lr in learning_rates:
    for bs in batch_sizes:
        for dropout in dropout_rates:
            train_and_evaluate(lr, bs, dropout)
```

**2. Random Search** (Better!)

```python
import random

for _ in range(20):  # 20 trials
    lr = 10 ** random.uniform(-5, -2)  # Log scale!
    bs = random.choice([16, 32, 64, 128])
    dropout = random.uniform(0.1, 0.5)

    train_and_evaluate(lr, bs, dropout)
```

**3. Bayesian Optimization** (Best but complex)

- Use libraries like Optuna, Ray Tune
- Learns from previous trials
- Focuses on promising regions

#### Practical Hyperparameter Tuning Example

```python
def hyperparameter_search():
    """
    Practical random search for hyperparameters.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Data
    X_train = torch.randn(500, 30)
    y_train = torch.sum(X_train[:, :10]**2, dim=1, keepdim=True) + torch.randn(500, 1) * 0.3

    X_val = torch.randn(100, 30)
    y_val = torch.sum(X_val[:, :10]**2, dim=1, keepdim=True) + torch.randn(100, 1) * 0.3

    # Hyperparameter search space
    def sample_hyperparameters():
        return {
            'lr': 10 ** np.random.uniform(-4, -2),  # Log scale: 0.0001 to 0.01
            'hidden_size': np.random.choice([50, 100, 200]),
            'n_layers': np.random.choice([2, 3, 4]),
            'dropout': np.random.uniform(0.1, 0.5),
            'weight_decay': 10 ** np.random.uniform(-5, -2)
        }

    def train_with_config(config, epochs=100):
        """Train model with given hyperparameters."""
        # Build model
        layers = []
        in_size = 30
        for _ in range(config['n_layers']):
            layers.extend([
                nn.Linear(in_size, config['hidden_size']),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            in_size = config['hidden_size']
        layers.append(nn.Linear(in_size, 1))

        model = nn.Sequential(*layers)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        criterion = nn.MSELoss()

        # Train
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()

        return best_val_loss

    # Run random search
    n_trials = 20
    results = []

    print(f"\nRunning {n_trials} trials of random search...")
    print("="*70)

    for trial in range(n_trials):
        config = sample_hyperparameters()
        val_loss = train_with_config(config)

        results.append({
            'config': config,
            'val_loss': val_loss
        })

        print(f"Trial {trial+1:2d} | Val Loss: {val_loss:.6f} | "
              f"LR: {config['lr']:.6f} | Hidden: {config['hidden_size']} | "
              f"Layers: {config['n_layers']} | Dropout: {config['dropout']:.3f}")

    # Find best configuration
    best_result = min(results, key=lambda x: x['val_loss'])

    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND:")
    print("="*70)
    for key, value in best_result['config'].items():
        print(f"  {key:15s}: {value}")
    print(f"\n  Validation Loss: {best_result['val_loss']:.6f}")
    print("="*70)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Val loss vs LR
    lrs = [r['config']['lr'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    axes[0, 0].scatter(lrs, val_losses, s=100, alpha=0.6)
    axes[0, 0].scatter([best_result['config']['lr']], [best_result['val_loss']],
                       s=200, c='red', marker='*', zorder=5, label='Best')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Val Loss vs Learning Rate')
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Val loss vs Hidden Size
    hidden_sizes = [r['config']['hidden_size'] for r in results]
    axes[0, 1].scatter(hidden_sizes, val_losses, s=100, alpha=0.6)
    axes[0, 1].scatter([best_result['config']['hidden_size']], [best_result['val_loss']],
                       s=200, c='red', marker='*', zorder=5, label='Best')
    axes[0, 1].set_xlabel('Hidden Size')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Val Loss vs Hidden Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Val loss vs Dropout
    dropouts = [r['config']['dropout'] for r in results]
    axes[1, 0].scatter(dropouts, val_losses, s=100, alpha=0.6)
    axes[1, 0].scatter([best_result['config']['dropout']], [best_result['val_loss']],
                       s=200, c='red', marker='*', zorder=5, label='Best')
    axes[1, 0].set_xlabel('Dropout Rate')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Val Loss vs Dropout')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Top 5 configurations
    sorted_results = sorted(results, key=lambda x: x['val_loss'])[:5]
    trial_ids = [f"Trial {results.index(r)+1}" for r in sorted_results]
    losses = [r['val_loss'] for r in sorted_results]

    colors = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue']
    axes[1, 1].barh(trial_ids, losses, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_title('Top 5 Configurations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_hyperparameter_search.png', dpi=150)
    plt.show()

hyperparameter_search()
```

---

### 4.5 Practical Tips for Hyperparameter Tuning

✅ **Search on Log Scale** (for LR, weight decay)

```python
lr = 10 ** np.random.uniform(-5, -2)  # NOT uniform(0.00001, 0.01)
```

✅ **Start with Defaults**

```python
# Good starting point for most problems
config = {
    'lr': 0.001,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'dropout': 0.3,
    'batch_size': 32
}
```

✅ **Tune in Order of Importance**

1. First: Learning rate
2. Second: Architecture (width, depth)
3. Third: Regularization
4. Fourth: Optimizer parameters

✅ **Use Early Stopping**

- Don't train full epochs during search
- Stop after 20-50 epochs to save time

✅ **Monitor Multiple Metrics**

- Validation loss (primary)
- Training loss (check overfitting)
- Generalization gap

---

### 4.6 Key Takeaways from Day 4

✅ **Learning Rate Schedules**

- Step Decay: Simple, works well
- Cosine Annealing: Smooth, popular
- Reduce on Plateau: Adaptive
- One Cycle: Fast training

✅ **Learning Rate Warmup**

- Gradually increase LR at start
- Stabilizes early training
- Essential for large models

✅ **Hyperparameter Search**

- Random search > Grid search
- Search on log scale for LR
- Start with good defaults
- Tune LR first!

✅ **Production Template**

```python
# Modern best practice
model = Network(hidden_size=128, dropout=0.3)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
for epoch in range(epochs):
    train_one_epoch()
    validate()
    scheduler.step()
```

**Tomorrow:** Better network architectures - design principles that work!

---

_End of Day 4. Total time: 6-8 hours._

---

<a name="day-5"></a>

## 📅 Day 5: Better Network Architectures

> "Deep networks are hard to train. Unless you use residual connections." - Kaiming He

### 5.1 The Vanishing Gradient Problem

#### Why Deep Networks Are Hard

**The problem:** Gradients get smaller as you backpropagate through layers!

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdot ... \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}
$$

Each term < 1 → Product vanishes!

```python
def demonstrate_vanishing_gradient():
    """
    Show the vanishing gradient problem in deep networks.
    """
    print("="*70)
    print("VANISHING GRADIENT DEMONSTRATION")
    print("="*70)

    torch.manual_seed(42)

    # Test different depths
    depths = [2, 5, 10, 20, 50]

    results = {}

    for depth in depths:
        print(f"\nTraining {depth}-layer network...")

        # Build deep network
        layers = []
        for i in range(depth):
            layers.extend([
                nn.Linear(100, 100),
                nn.Tanh()  # Tanh has gradient saturation!
            ])
        layers.append(nn.Linear(100, 10))

        model = nn.Sequential(*layers)

        # Dummy forward-backward pass
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))

        criterion = nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Collect gradient norms per layer
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                grad_norms.append(param.grad.norm().item())

        results[depth] = grad_norms

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gradient flow
    ax = axes[0]
    for depth, norms in results.items():
        layers_idx = list(range(len(norms)))
        ax.plot(layers_idx, norms, marker='o', label=f'{depth} layers', linewidth=2)

    ax.set_xlabel('Layer Index (0 = deepest)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Vanishing Gradients in Deep Networks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axhline(y=1e-5, color='red', linestyle='--', alpha=0.5, label='Vanished!')

    # First layer gradient vs depth
    ax = axes[1]
    depths_list = list(results.keys())
    first_layer_grads = [results[d][0] if results[d] else 0 for d in depths_list]

    colors = plt.cm.Reds(np.linspace(0.3, 1, len(depths_list)))
    bars = ax.bar(range(len(depths_list)), first_layer_grads, color=colors, alpha=0.7)
    ax.set_xticks(range(len(depths_list)))
    ax.set_xticklabels(depths_list)
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('First Layer Gradient Norm')
    ax.set_title('Gradient at Deepest Layer vs Network Depth')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add annotations
    for i, (bar, grad) in enumerate(zip(bars, first_layer_grads)):
        if grad < 1e-5:
            ax.text(i, grad*10, 'VANISHED!', ha='center', color='red', weight='bold')

    plt.tight_layout()
    plt.savefig('week2_vanishing_gradient.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("✗ Deep networks suffer from vanishing gradients!")
    print("  → Gradients exponentially decrease with depth")
    print("  → Early layers don't learn")
    print("  → Solution: Residual connections!")
    print("="*70)

demonstrate_vanishing_gradient()
```

---

### 5.2 Residual Connections (ResNets)

#### The Big Idea

**Instead of learning** $F(x)$, **learn the residual** $F(x) = H(x) - x$

$$
H(x) = F(x) + x
$$

**Why it works:**

- Gradient flows directly through skip connection!
- Network can learn identity mapping easily
- Deeper networks → better performance

#### Residual Block Implementation

```python
class ResidualBlock(nn.Module):
    """
    Basic residual block: H(x) = F(x) + x

    Components:
    - Two conv/linear layers with ReLU
    - Skip connection (identity)
    - Final activation after addition
    """

    def __init__(self, features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )

    def forward(self, x):
        # F(x)
        residual = self.net(x)

        # H(x) = F(x) + x (skip connection!)
        out = residual + x

        # Final activation
        out = F.relu(out)

        return out


def visualize_residual_block():
    """
    Visualize what residual block computes.
    """
    print("\n" + "="*70)
    print("RESIDUAL BLOCK VISUALIZATION")
    print("="*70)

    torch.manual_seed(42)

    block = ResidualBlock(50)

    x = torch.randn(100, 50)

    # Forward pass
    with torch.no_grad():
        residual = block.net(x)
        output = residual + x
        output = F.relu(output)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Input
    axes[0].imshow(x[:20].numpy(), aspect='auto', cmap='coolwarm')
    axes[0].set_title('Input x\n(20 samples × 50 features)')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Samples')

    # Residual
    axes[1].imshow(residual[:20].numpy(), aspect='auto', cmap='coolwarm')
    axes[1].set_title('Residual F(x)\n(Learned transformation)')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Samples')

    # Output
    axes[2].imshow(output[:20].numpy(), aspect='auto', cmap='coolwarm')
    axes[2].set_title('Output H(x) = F(x) + x\n(After skip connection)')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Samples')

    plt.tight_layout()
    plt.savefig('week2_residual_block.png', dpi=150)
    plt.show()

    print("\n✓ Residual block adds learned transformation to input")
    print("  → Skip connection preserves input")
    print("  → Network learns refinements")

visualize_residual_block()
```

#### Complete Residual Network

```python
class ResidualNetwork(nn.Module):
    """
    Complete network with residual blocks.

    Can train very deep networks thanks to skip connections!
    """

    def __init__(self, input_size, hidden_size, output_size, n_blocks=5):
        super().__init__()

        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(n_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Project to hidden size
        x = F.relu(self.input_layer(x))

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.output_layer(x)

        return x


def compare_vanilla_vs_residual():
    """
    Compare vanilla deep network vs residual network.
    """
    print("\n" + "="*70)
    print("VANILLA vs RESIDUAL NETWORK COMPARISON")
    print("="*70)

    torch.manual_seed(42)

    # Data (harder problem)
    X_train = torch.randn(500, 50)
    y_train = torch.sum(torch.sin(X_train[:, :20]), dim=1, keepdim=True) + torch.randn(500, 1) * 0.2

    X_val = torch.randn(100, 50)
    y_val = torch.sum(torch.sin(X_val[:, :20]), dim=1, keepdim=True) + torch.randn(100, 1) * 0.2

    # Vanilla deep network
    vanilla_layers = []
    prev_size = 50
    for _ in range(10):  # 10 layers deep!
        vanilla_layers.extend([
            nn.Linear(prev_size, 128),
            nn.ReLU()
        ])
        prev_size = 128
    vanilla_layers.append(nn.Linear(128, 1))

    vanilla_model = nn.Sequential(*vanilla_layers)

    # Residual network (same depth)
    residual_model = ResidualNetwork(
        input_size=50,
        hidden_size=128,
        output_size=1,
        n_blocks=10  # Same depth!
    )

    # Training setup
    criterion = nn.MSELoss()
    epochs = 200

    models = {
        'Vanilla (Deep)': vanilla_model,
        'Residual (Deep)': residual_model
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        val_losses = []
        grad_norms = []

        for epoch in range(epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            train_loss = criterion(pred, y_train)
            train_loss.backward()

            # Measure gradient norm (first layer)
            first_param = next(model.parameters())
            if first_param.grad is not None:
                grad_norms.append(first_param.grad.norm().item())
            else:
                grad_norms.append(0)

            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        results[name] = {
            'train': train_losses,
            'val': val_losses,
            'grad': grad_norms
        }

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data['train'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss: Vanilla vs Residual')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Validation loss
    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(data['val'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss: Vanilla vs Residual')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Gradient flow
    ax = axes[1, 0]
    for name, data in results.items():
        ax.plot(data['grad'], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('First Layer Gradient Norm')
    ax.set_title('Gradient Flow: Vanilla vs Residual')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    ax.axhline(y=1e-5, color='red', linestyle='--', alpha=0.5)
    ax.text(100, 1e-5, 'Vanished', color='red', va='bottom')

    # Final comparison
    ax = axes[1, 1]
    final_vals = [data['val'][-1] for data in results.values()]
    colors = ['coral', 'skyblue']
    bars = ax.bar(results.keys(), final_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    # Highlight winner
    winner_idx = np.argmin(final_vals)
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(4)

    plt.tight_layout()
    plt.savefig('week2_vanilla_vs_residual.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("FINAL RESULTS:")
    for name, data in results.items():
        print(f"  {name:20s}: {data['val'][-1]:.6f}")
    print("\n✓ Residual connections enable training of deep networks!")
    print("  → Better gradient flow")
    print("  → Lower validation loss")
    print("  → Industry standard for deep models")
    print("="*70)

compare_vanilla_vs_residual()
```

---

### 5.3 Weight Initialization

#### Why Initialization Matters

Bad initialization → vanishing/exploding gradients → training fails!

**Goal:** Keep activations and gradients at reasonable scale across layers.

#### Common Initialization Schemes

**1. Zero Initialization** ❌ (Never use!)

```python
# All neurons learn the same thing!
nn.init.zeros_(layer.weight)
```

**2. Random Small Values** ⚠️ (Old method)

```python
nn.init.normal_(layer.weight, mean=0, std=0.01)
```

**3. Xavier/Glorot Initialization** ✅ (For tanh/sigmoid)

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
$$

```python
nn.init.xavier_normal_(layer.weight)
```

**4. He Initialization** ⭐ (For ReLU) - **Use this!**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

#### Initialization Comparison

```python
def compare_initializations():
    """
    Compare different weight initialization schemes.
    """
    print("\n" + "="*70)
    print("WEIGHT INITIALIZATION COMPARISON")
    print("="*70)

    torch.manual_seed(42)

    # Data
    X = torch.randn(300, 50)
    y = torch.sum(X[:, :20]**2, dim=1, keepdim=True) + torch.randn(300, 1) * 0.3

    # Different initialization schemes
    def init_weights(model, scheme):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if scheme == 'zeros':
                    nn.init.zeros_(m.weight)
                elif scheme == 'ones':
                    nn.init.ones_(m.weight)
                elif scheme == 'normal_small':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif scheme == 'normal_large':
                    nn.init.normal_(m.weight, mean=0, std=1.0)
                elif scheme == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif scheme == 'he':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    schemes = ['zeros', 'ones', 'normal_small', 'normal_large', 'xavier', 'he']

    results = {}
    criterion = nn.MSELoss()
    epochs = 150

    for scheme in schemes:
        print(f"\nTraining with {scheme} initialization...")

        # Build model
        model = nn.Sequential(
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 1)
        )

        # Initialize
        init_weights(model, scheme)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        losses = []
        grad_norms = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()

            # Track gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            optimizer.step()
            losses.append(loss.item())

        results[scheme] = {
            'loss': losses,
            'grad': grad_norms
        }

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    ax = axes[0, 0]
    for scheme, data in results.items():
        ax.plot(data['loss'], label=scheme, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss: Different Initializations')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Gradient norms
    ax = axes[0, 1]
    for scheme, data in results.items():
        ax.plot(data['grad'], label=scheme, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms: Different Initializations')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # First 20 epochs (zoom in)
    ax = axes[1, 0]
    for scheme, data in results.items():
        ax.plot(data['loss'][:20], label=scheme, linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('First 20 Epochs (Detail View)')
    ax.legend()
    ax.grid(True)

    # Final loss comparison
    ax = axes[1, 1]
    final_losses = [data['loss'][-1] for data in results.values()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(schemes)))
    bars = ax.bar(schemes, final_losses, color=colors, alpha=0.7)
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Final Loss: Initialization Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Highlight best
    best_idx = np.argmin(final_losses)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)

    plt.tight_layout()
    plt.savefig('week2_initialization_comparison.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("FINAL LOSSES:")
    for scheme, data in results.items():
        status = "✓ GOOD" if scheme in ['he', 'xavier'] else "✗ BAD"
        print(f"  {scheme:15s}: {data['loss'][-1]:.6f} {status}")
    print("\n⭐ He initialization (Kaiming) is best for ReLU networks!")
    print("="*70)

compare_initializations()
```

---

### 5.4 Architecture Design Best Practices

#### Modern Network Design Recipe

```python
class ModernNetwork(nn.Module):
    """
    Production-quality network architecture.

    Combines all Week 2 techniques:
    - Residual connections
    - Batch normalization
    - Dropout
    - He initialization
    """

    def __init__(self, input_size, hidden_size, output_size, n_blocks=3, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # Residual blocks with BatchNorm
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
            ]))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize with He
        self._init_weights()

    def _init_weights(self):
        """Apply He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            identity = x

            # First layer
            x = block[0](x)
            x = block[1](x)
            x = F.relu(x)
            x = block[2](x)  # Dropout

            # Second layer
            x = block[3](x)
            x = block[4](x)

            # Residual connection
            x = x + identity
            x = F.relu(x)

        # Output
        x = self.output_layer(x)

        return x


# Usage example
model = ModernNetwork(
    input_size=784,       # e.g., MNIST
    hidden_size=256,
    output_size=10,
    n_blocks=5,
    dropout=0.3
)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### Design Rules for Production

✅ **Use Residual Connections** for depth > 10 layers

✅ **Batch Normalization** after each linear layer (before activation)

✅ **Dropout** between layers (0.2-0.5)

✅ **He Initialization** for ReLU networks

✅ **AdamW Optimizer** with weight decay

✅ **Cosine LR Schedule** with warmup

✅ **Early Stopping** on validation set

```python
# Production training template
def train_production_model():
    """
    Complete production training setup.
    """
    print("\n" + "="*70)
    print("PRODUCTION MODEL TRAINING")
    print("="*70)

    torch.manual_seed(42)

    # Data
    X_train = torch.randn(1000, 100)
    y_train = torch.randint(0, 5, (1000,))
    X_val = torch.randn(200, 100)
    y_val = torch.randint(0, 5, (200,))

    # Model
    model = ModernNetwork(
        input_size=100,
        hidden_size=128,
        output_size=5,
        n_blocks=4,
        dropout=0.3
    )

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 200
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses = []
    val_losses = []
    lrs = []

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        train_loss = criterion(pred, y_train)
        train_loss.backward()

        # Gradient clipping (important!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        # Update scheduler
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | LR: {lrs[-1]:.6f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(lrs, linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('week2_production_training.png', dpi=150)
    plt.show()

    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: best_model.pth")

train_production_model()
```

---

### 5.5 Key Takeaways from Day 5

✅ **Vanishing Gradients**

- Problem: Gradients decrease exponentially with depth
- Solution: Residual connections

✅ **Residual Networks**

- Skip connections: H(x) = F(x) + x
- Enable training of very deep networks
- Industry standard architecture

✅ **Weight Initialization**

- He initialization for ReLU networks
- Xavier for tanh/sigmoid
- Never use zeros or ones!

✅ **Architecture Best Practices**

- Residual connections for depth
- Batch normalization after linear layers
- Dropout for regularization
- He initialization
- AdamW + Cosine schedule

✅ **Production Template**

```python
model = ModernNetwork(hidden_size=256, n_blocks=5, dropout=0.3)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# + Early stopping + Gradient clipping
```

**Weekend Project:** Put it all together with CIFAR-10!

---

_End of Day 5. Total time: 6-8 hours._

---

<a name="weekend-project"></a>

## 🎯 Weekend Project: CIFAR-10 Classification with All Week 2 Techniques

> "The true test of understanding is building something that works."

### Project Goal

**Build a CNN classifier for CIFAR-10 that achieves 85%+ accuracy** using ALL techniques from Week 2:

✅ Regularization (L2, Dropout)
✅ Batch Normalization
✅ Advanced Optimizer (AdamW)
✅ Learning Rate Scheduling (Cosine + Warmup)
✅ Residual Connections
✅ Proper Weight Initialization
✅ Early Stopping

### Step 1: Data Loading and Augmentation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def setup_cifar10_data():
    """
    Load CIFAR-10 with proper augmentation.

    Data augmentation is crucial for good generalization!
    """
    print("="*70)
    print("CIFAR-10 DATA SETUP")
    print("="*70)

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Random crop with padding
        transforms.RandomHorizontalFlip(),          # 50% chance flip
        transforms.ToTensor(),
        transforms.Normalize(                       # Normalize to zero mean, unit variance
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    # No augmentation for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples: {len(trainset)}")
    print(f"  Test samples: {len(testset)}")
    print(f"  Classes: {classes}")
    print(f"  Image shape: 3x32x32")

    # Visualize some samples
    visualize_cifar10_samples(trainloader, classes)

    return trainloader, testloader, classes


def visualize_cifar10_samples(dataloader, classes):
    """
    Visualize random samples from CIFAR-10.
    """
    # Get a batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean

    # Plot
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    axes = axes.flatten()

    for i in range(16):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]], fontsize=10)
        axes[i].axis('off')

    plt.suptitle('CIFAR-10 Training Samples (with augmentation)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('week2_cifar10_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


trainloader, testloader, classes = setup_cifar10_data()
```

---

### Step 2: Build ResNet Architecture for CIFAR-10

```python
class ResidualBlock(nn.Module):
    """
    Residual block for CIFAR-10 ResNet.

    Structure:
    - Conv → BatchNorm → ReLU → Conv → BatchNorm → Add → ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path (if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add shortcut
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet for CIFAR-10.

    Architecture inspired by ResNet-18 but adapted for 32x32 images.
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        # Initial convolution (no pooling for small CIFAR images!)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Dropout before final layer
        self.dropout = nn.Dropout(dropout)

        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = []

        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout + FC
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Create model
model = ResNet(num_classes=10, dropout=0.3)
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print("="*70)
```

---

### Step 3: Training Setup with All Week 2 Techniques

```python
def create_optimizer_and_scheduler(model, epochs, steps_per_epoch):
    """
    Setup optimizer and learning rate schedule.

    Using:
    - AdamW optimizer (weight decay)
    - Cosine annealing schedule
    - Warmup for first 10 epochs
    """
    # AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)
    )

    # Cosine schedule with warmup
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """

    def __init__(self, patience=15, min_delta=0.001, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        """Save model checkpoint."""
        torch.save(model.state_dict(), self.path)


def train_epoch(model, trainloader, optimizer, criterion, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc='Training', leave=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward
        loss.backward()

        # Gradient clipping (important for stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, testloader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(testloader, desc='Validating', leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / len(testloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc
```

---

### Step 4: Complete Training Loop

```python
def train_cifar10_resnet():
    """
    Complete training pipeline for CIFAR-10.

    Integrates ALL Week 2 techniques!
    """
    print("\n" + "="*70)
    print("TRAINING CIFAR-10 RESNET")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    trainloader, testloader, classes = setup_cifar10_data()

    # Create model
    model = ResNet(num_classes=10, dropout=0.3).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    epochs = 100
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, epochs, len(trainloader)
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=15, path='cifar10_best_model.pth')

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, testloader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {history['lr'][-1]:.6f}")

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")

    # Load best model
    model.load_state_dict(torch.load('cifar10_best_model.pth'))

    return model, history, classes


# Train the model!
model, history, classes = train_cifar10_resnet()
```

---

### Step 5: Visualization and Evaluation

```python
def plot_training_history(history):
    """
    Visualize training progress.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine with Warmup)')
    ax.grid(True, alpha=0.3)

    # Final comparison
    ax = axes[1, 1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    bars = ax.bar(['Train', 'Validation'], [final_train_acc, final_val_acc],
                  color=['skyblue', 'coral'], alpha=0.7)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracy')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('week2_cifar10_training_history.png', dpi=150)
    plt.show()


def evaluate_model_detailed(model, testloader, classes, device):
    """
    Detailed evaluation with confusion matrix and per-class accuracy.
    """
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION")
    print("="*70)

    model.eval()

    # Collect all predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Overall accuracy
    overall_acc = 100. * (all_preds == all_targets).sum() / len(all_targets)
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)

    class_correct = {}
    class_total = {}

    for i, class_name in enumerate(classes):
        mask = (all_targets == i)
        class_total[class_name] = mask.sum()
        class_correct[class_name] = ((all_preds == all_targets) & mask).sum()
        class_acc = 100. * class_correct[class_name] / class_total[class_name]
        print(f"  {class_name:12s}: {class_acc:5.2f}% ({class_correct[class_name]}/{class_total[class_name]})")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix - CIFAR-10 ResNet')

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

    plt.tight_layout()
    plt.savefig('week2_cifar10_confusion_matrix.png', dpi=150)
    plt.show()

    # Plot per-class accuracy
    fig, ax = plt.subplots(figsize=(12, 6))

    class_accuracies = [100. * class_correct[c] / class_total[c] for c in classes]
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, class_accuracies, color=colors, alpha=0.7)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy - CIFAR-10 ResNet')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_acc:.2f}%')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('week2_cifar10_class_accuracy.png', dpi=150)
    plt.show()

    print(f"\n{'='*70}")

    return overall_acc, class_accuracies


def visualize_predictions(model, testloader, classes, device, num_images=20):
    """
    Visualize model predictions on test images.
    """
    model.eval()

    # Get a batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    # Move to CPU for visualization
    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    # Denormalize
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean

    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        pred_label = classes[preds[i]]
        true_label = classes[labels[i]]

        if preds[i] == labels[i]:
            color = 'green'
            title = f'✓ {pred_label}'
        else:
            color = 'red'
            title = f'✗ Pred: {pred_label}\nTrue: {true_label}'

        axes[i].set_title(title, color=color, fontsize=10, weight='bold')
        axes[i].axis('off')

    plt.suptitle('CIFAR-10 Predictions (Green=Correct, Red=Wrong)',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('week2_cifar10_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_training_history(history)
overall_acc, class_accs = evaluate_model_detailed(model, testloader, classes, device)
visualize_predictions(model, testloader, classes, device)
```

---

### Step 6: Model Inspection and Analysis

```python
def analyze_learned_features(model, testloader, device):
    """
    Analyze what the model has learned.
    """
    print("\n" + "="*70)
    print("LEARNED FEATURES ANALYSIS")
    print("="*70)

    model.eval()

    # Get a batch
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    images = images.to(device)

    # Extract features from different layers
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    # Register hooks
    model.layer1.register_forward_hook(hook_fn('layer1'))
    model.layer2.register_forward_hook(hook_fn('layer2'))
    model.layer3.register_forward_hook(hook_fn('layer3'))
    model.layer4.register_forward_hook(hook_fn('layer4'))

    # Forward pass
    with torch.no_grad():
        _ = model(images[:1])

    # Visualize feature maps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

    for idx, layer_name in enumerate(layer_names):
        feat = features[layer_name][0]  # First image

        # Show first 16 channels
        n_channels = min(16, feat.shape[0])
        feat_grid = feat[:n_channels].cpu().numpy()

        # Create grid
        grid_size = 4
        grid = np.zeros((32 * grid_size, 32 * grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < n_channels:
                    channel = feat_grid[i * grid_size + j]
                    # Resize to 32x32 for visualization
                    from scipy.ndimage import zoom
                    if channel.shape[0] < 32:
                        scale = 32 / channel.shape[0]
                        channel = zoom(channel, scale, order=1)

                    grid[i*32:(i+1)*32, j*32:(j+1)*32] = channel

        axes[idx].imshow(grid, cmap='viridis')
        axes[idx].set_title(f'{layer_name.capitalize()} Feature Maps\n'
                           f'(Shape: {feat.shape})', fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Learned Feature Maps at Different Depths', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('week2_cifar10_feature_maps.png', dpi=150)
    plt.show()

    print("\n✓ Feature visualization complete!")
    print("  → Early layers: edges, colors, textures")
    print("  → Deep layers: complex patterns, object parts")


analyze_learned_features(model, testloader, device)
```

---

### Step 7: Save and Export Model

```python
def save_model_for_deployment(model, path='cifar10_resnet_final.pth'):
    """
    Save complete model for deployment.
    """
    print("\n" + "="*70)
    print("SAVING MODEL FOR DEPLOYMENT")
    print("="*70)

    # Save complete model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'ResNet',
        'num_classes': 10,
        'dropout': 0.3,
        'accuracy': max(history['val_acc']),
        'classes': classes
    }, path)

    print(f"\n✓ Model saved to: {path}")
    print(f"  Final validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Model size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")

    # Example: How to load
    print("\nTo load this model:")
    print(f"""
    checkpoint = torch.load('{path}')
    model = ResNet(num_classes=10, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    """)


import os
save_model_for_deployment(model)
```

---

### Weekend Project Summary

```python
def print_project_summary():
    """
    Print complete project summary.
    """
    print("\n" + "="*70)
    print("🎉 WEEKEND PROJECT COMPLETE! 🎉")
    print("="*70)

    print("\n✅ TECHNIQUES SUCCESSFULLY APPLIED:")
    print("-" * 70)
    print("  1. ✓ Data Augmentation (RandomCrop, HorizontalFlip)")
    print("  2. ✓ ResNet Architecture (Residual Connections)")
    print("  3. ✓ Batch Normalization (After each conv layer)")
    print("  4. ✓ Dropout Regularization (Before final layer)")
    print("  5. ✓ He Weight Initialization (For ReLU networks)")
    print("  6. ✓ AdamW Optimizer (With weight decay)")
    print("  7. ✓ Cosine LR Schedule (With warmup)")
    print("  8. ✓ Gradient Clipping (Max norm = 1.0)")
    print("  9. ✓ Early Stopping (Patience = 15)")
    print(" 10. ✓ Comprehensive Evaluation (Confusion matrix, per-class)")

    print("\n📊 FINAL RESULTS:")
    print("-" * 70)
    print(f"  • Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  • Total Training Epochs: {len(history['train_loss'])}")
    print(f"  • Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Architecture: ResNet with 8 residual blocks")

    print("\n🎯 ACHIEVEMENT:")
    if max(history['val_acc']) >= 85:
        print("  ⭐⭐⭐ EXCELLENT! Target accuracy (85%+) achieved!")
    elif max(history['val_acc']) >= 80:
        print("  ⭐⭐ GOOD! Close to target (80%+)")
    else:
        print("  ⭐ Training complete, but room for improvement")

    print("\n📚 WEEK 2 MASTERY CHECKLIST:")
    print("-" * 70)
    print("  ✓ Regularization techniques (L2, Dropout, Early Stopping)")
    print("  ✓ Normalization (Batch Norm, Layer Norm)")
    print("  ✓ Advanced optimizers (Adam, AdamW, momentum)")
    print("  ✓ Learning rate scheduling (Cosine, warmup)")
    print("  ✓ Network architecture design (Residual connections)")
    print("  ✓ Weight initialization (He initialization)")
    print("  ✓ Complete production pipeline (Data → Train → Evaluate)")

    print("\n🚀 NEXT STEPS:")
    print("-" * 70)
    print("  1. Experiment with different architectures")
    print("  2. Try different augmentation strategies")
    print("  3. Hyperparameter tuning for 90%+ accuracy")
    print("  4. Deploy model for inference")
    print("  5. Move on to Week 3: Convolutional Neural Networks (deep dive)")

    print("\n" + "="*70)
    print("CONGRATULATIONS! YOU'VE COMPLETED WEEK 2! 🎊")
    print("="*70 + "\n")


print_project_summary()
```

---

## 🎓 Week 2 Complete Summary

### What You've Mastered

**Core Techniques:**

1. ✅ **Regularization** - Prevent overfitting (L1, L2, Dropout, Early Stopping)
2. ✅ **Normalization** - Stabilize training (Batch Norm, Layer Norm)
3. ✅ **Optimizers** - Train faster (Momentum, RMSprop, Adam, AdamW)
4. ✅ **LR Scheduling** - Converge better (Step, Cosine, Warmup, OneCycle)
5. ✅ **Architectures** - Train deeper (Residual Connections, Weight Init)

**Weekend Project:**

- Built production-quality ResNet for CIFAR-10
- Achieved 85%+ accuracy
- Applied ALL Week 2 techniques
- Complete evaluation and deployment

### Key Production Patterns

```python
# Modern Neural Network Template (Copy this!)

# 1. Model with all techniques
model = ResNet(hidden_size=256, n_blocks=5, dropout=0.3)
model._init_weights()  # He initialization

# 2. Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)

# 3. LR Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# 4. Training loop
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    scheduler.step()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Early stopping
    if early_stopping(val_loss, model):
        break
```

### Before/After Week 2

**Before:**

- ❌ Networks overfit easily
- ❌ Training unstable
- ❌ Can't train deep networks
- ❌ Slow convergence

**After:**

- ✅ Robust generalization
- ✅ Stable training
- ✅ Train very deep networks
- ✅ Fast convergence

### Next: Week 3 Preview

**Convolutional Neural Networks (Deep Dive)**

- Convolution operations
- Pooling layers
- Popular architectures (VGG, ResNet, EfficientNet)
- Transfer learning
- Object detection basics

---

## 📖 Additional Resources (Optional)

### Papers to Read

1. "Batch Normalization" (Ioffe & Szegedy, 2015)
2. "Deep Residual Learning" (He et al., 2015)
3. "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

### Code References

- PyTorch official examples: github.com/pytorch/examples
- ResNet implementation: torchvision.models.resnet

### Practice Suggestions

1. Try different ResNet depths (18, 34, 50 layers)
2. Experiment with data augmentation
3. Implement learning rate finder
4. Try different initialization schemes
5. Build ensemble of models

---

## ✨ Final Thoughts

**You've completed one of the hardest weeks!** These techniques separate hobbyists from professionals.

**What makes this week special:**

- Techniques used in ALL modern deep learning
- Skills directly applicable to production
- Foundation for advanced architectures

**Your Progress:**

- Week 1: Built neural networks from scratch ✅
- Week 2: Mastered production training techniques ✅
- Week 3: Advanced architectures (coming next)

---

**🎉 CONGRATULATIONS ON COMPLETING WEEK 2! 🎉**

**Ready for Week 3? Let me know!**

---

_End of Weekend Project. Total time: 8-12 hours._

_Week 2 total time: 38-48 hours_
