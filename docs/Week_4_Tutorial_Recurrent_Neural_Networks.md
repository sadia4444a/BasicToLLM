# Week 4: Recurrent Neural Networks - Complete Tutorial

> **Expert Guide by a Senior ML Engineer with 15+ Years Experience**
>
> This tutorial covers Recurrent Neural Networks (RNNs) from fundamental principles to production deployment. Everything you need is here—no external resources required.

---

## 📋 Table of Contents

1. [Introduction: Why RNNs?](#introduction)
2. [Day 1: RNN Fundamentals](#day-1)
3. [Day 2: LSTM and GRU](#day-2)
4. [Day 3: Advanced RNN Architectures](#day-3)
5. [Day 4: Attention Mechanisms](#day-4)
6. [Day 5: Transformers and Modern Architectures](#day-5)
7. [Weekend Project: Text Generation Engine](#weekend-project)
8. [Week Review](#week-review)

---

<a name="introduction"></a>

## 🎯 Introduction: Why Recurrent Neural Networks?

### The Problem with Feedforward Networks

**So far:** We've learned about:

- **Week 1:** Fully connected networks (feedforward)
- **Week 2:** Training techniques (optimization, regularization)
- **Week 3:** Convolutional networks (spatial patterns in images)

**Limitation:** None of these handle **sequences**!

### What Are Sequences?

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Sequence examples
sequences = {
    'Text': "Hello, how are you?",
    'Time Series': [23.5, 24.1, 23.8, 24.5, 25.2],  # Temperature readings
    'Audio': [0.1, 0.3, 0.5, 0.4, 0.2],  # Sound wave samples
    'Video': ["Frame 1", "Frame 2", "Frame 3"],  # Sequence of images
    'DNA': "ATCGGTACG"  # Genetic sequence
}

print("="*70)
print("EXAMPLES OF SEQUENTIAL DATA")
print("="*70)

for data_type, example in sequences.items():
    print(f"\n{data_type}:")
    print(f"  {example}")
    print(f"  → Order matters! Can't shuffle without losing meaning")

print("\n" + "="*70)
print("KEY PROPERTY: TEMPORAL DEPENDENCIES")
print("="*70)
print("""
Each element depends on previous elements:
  • "I like apples" ≠ "apples like I"
  • Temperature at t depends on temperature at t-1
  • Next word depends on previous words

→ We need networks with MEMORY!
""")
```

### Why CNNs Don't Work for Sequences

```python
def compare_cnn_vs_rnn():
    """
    Visualize why CNNs aren't ideal for sequences.
    """
    print("\n" + "="*70)
    print("CNN vs RNN: ARCHITECTURAL DIFFERENCES")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CNN Architecture
    axes[0].text(0.5, 0.9, 'CNN: Fixed-Size Input', ha='center', fontsize=14, weight='bold')
    axes[0].text(0.5, 0.75, 'Input: 224×224 image', ha='center', fontsize=11)
    axes[0].arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    axes[0].text(0.5, 0.5, 'Conv Layers\n(Spatial Features)', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[0].arrow(0.5, 0.35, 0, -0.1, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    axes[0].text(0.5, 0.2, 'Output: Class', ha='center', fontsize=11)
    axes[0].text(0.5, 0.05, '❌ Cannot handle variable-length input', ha='center', fontsize=10, color='red')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')

    # RNN Architecture
    axes[1].text(0.5, 0.9, 'RNN: Variable-Length Input', ha='center', fontsize=14, weight='bold')

    # Show sequence
    x_positions = np.linspace(0.2, 0.8, 5)
    for i, x in enumerate(x_positions):
        axes[1].add_patch(plt.Circle((x, 0.7), 0.03, color='green', alpha=0.6))
        axes[1].text(x, 0.62, f'x{i+1}', ha='center', fontsize=9)

    axes[1].text(0.5, 0.75, '← Input Sequence (any length) →', ha='center', fontsize=10, style='italic')

    # RNN cells
    for i, x in enumerate(x_positions):
        axes[1].add_patch(plt.Rectangle((x-0.04, 0.45), 0.08, 0.08,
                                        facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
        axes[1].text(x, 0.49, 'RNN', ha='center', fontsize=8, weight='bold')

        # Connections
        if i < len(x_positions) - 1:
            axes[1].arrow(x+0.04, 0.49, x_positions[i+1]-x-0.08, 0,
                         head_width=0.02, head_length=0.02, fc='purple', ec='purple', linewidth=2)
            axes[1].text((x+x_positions[i+1])/2, 0.52, 'h', ha='center', fontsize=8, color='purple', weight='bold')

    axes[1].text(0.5, 0.35, '↓', ha='center', fontsize=16)
    axes[1].text(0.5, 0.25, 'Output: Sequence or Single Value', ha='center', fontsize=11)
    axes[1].text(0.5, 0.15, '✅ Handles any sequence length', ha='center', fontsize=10, color='green', weight='bold')
    axes[1].text(0.5, 0.08, '✅ Maintains memory across time', ha='center', fontsize=10, color='green', weight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('week4_cnn_vs_rnn.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n📊 KEY DIFFERENCES:")
    print("-" * 70)
    print(f"{'Property':<25} {'CNN':<20} {'RNN':<20}")
    print("-" * 70)
    print(f"{'Input Size':<25} {'Fixed':<20} {'Variable':<20}")
    print(f"{'Best For':<25} {'Spatial Data':<20} {'Sequential Data':<20}")
    print(f"{'Memory':<25} {'None':<20} {'Hidden State':<20}")
    print(f"{'Example':<25} {'Images':<20} {'Text, Audio':<20}")
    print("-" * 70)

    print("\n✓ RNNs process sequences one element at a time, maintaining memory!")

compare_cnn_vs_rnn()
```

### RNN Applications

```python
def show_rnn_applications():
    """
    Real-world RNN applications.
    """
    print("\n" + "="*70)
    print("REAL-WORLD RNN APPLICATIONS")
    print("="*70)

    applications = {
        'Natural Language Processing': [
            'Machine Translation (English → French)',
            'Text Generation (GPT-style)',
            'Sentiment Analysis',
            'Named Entity Recognition',
            'Question Answering'
        ],
        'Time Series Analysis': [
            'Stock Price Prediction',
            'Weather Forecasting',
            'Energy Consumption Prediction',
            'Anomaly Detection',
            'Demand Forecasting'
        ],
        'Speech & Audio': [
            'Speech Recognition (Siri, Alexa)',
            'Music Generation',
            'Speech Synthesis (Text-to-Speech)',
            'Audio Classification',
            'Voice Cloning'
        ],
        'Video Analysis': [
            'Action Recognition',
            'Video Captioning',
            'Video Prediction',
            'Gesture Recognition',
            'Video Summarization'
        ],
        'Healthcare': [
            'Patient Trajectory Prediction',
            'Medical Report Generation',
            'Drug Discovery',
            'ECG Analysis',
            'Clinical Notes Analysis'
        ]
    }

    for category, apps in applications.items():
        print(f"\n🔹 {category}:")
        for app in apps:
            print(f"   • {app}")

    print("\n✓ RNNs power most modern NLP and time series applications!")

show_rnn_applications()
```

### This Week's Journey

```python
def visualize_week_roadmap():
    """
    Visualize Week 4 learning roadmap.
    """
    print("\n" + "="*70)
    print("WEEK 4 LEARNING ROADMAP")
    print("="*70)

    roadmap = {
        'Day 1': {
            'topic': 'RNN Fundamentals',
            'content': ['Vanilla RNN', 'Backpropagation Through Time', 'Vanishing Gradients']
        },
        'Day 2': {
            'topic': 'LSTM and GRU',
            'content': ['LSTM Architecture', 'GRU Simplified', 'Solving Vanishing Gradients']
        },
        'Day 3': {
            'topic': 'Advanced Architectures',
            'content': ['Bidirectional RNNs', 'Encoder-Decoder', 'Seq2Seq Models']
        },
        'Day 4': {
            'topic': 'Attention Mechanisms',
            'content': ['Attention Basics', 'Self-Attention', 'Multi-Head Attention']
        },
        'Day 5': {
            'topic': 'Transformers',
            'content': ['Transformer Architecture', 'BERT & GPT', 'Modern NLP']
        },
        'Weekend': {
            'topic': 'Text Generation',
            'content': ['Character-Level RNN', 'Word-Level LSTM', 'Complete Generator']
        }
    }

    fig, ax = plt.subplots(figsize=(16, 10))

    y_pos = 0.9
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(roadmap)))

    for idx, (day, info) in enumerate(roadmap.items()):
        # Day title
        ax.text(0.05, y_pos, day, fontsize=13, weight='bold', color=colors[idx])
        ax.text(0.15, y_pos, info['topic'], fontsize=12, weight='bold')

        # Content
        for i, item in enumerate(info['content']):
            ax.text(0.2, y_pos - 0.05 - i*0.03, f'• {item}', fontsize=10)

        y_pos -= 0.15

        if y_pos > 0.05:
            ax.plot([0.05, 0.95], [y_pos + 0.02, y_pos + 0.02], 'k--', alpha=0.3, linewidth=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Week 4: Recurrent Neural Networks - Complete Roadmap',
                fontsize=16, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('week4_roadmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✓ By the end of this week, you'll master sequence modeling!")
    print("  → From basic RNNs to state-of-the-art Transformers")
    print("  → Build a text generation engine")
    print("  → Understand GPT and BERT architectures")

visualize_week_roadmap()
```

**Let's begin!** 🚀

---

<a name="day-1"></a>

## 📅 Day 1: RNN Fundamentals

> "The future depends on what you do today." - Mahatma Gandhi

### 1.1 The Basic RNN Cell

**Core Idea:** Maintain a **hidden state** that gets updated at each time step

**Mathematical Formula:**

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

Where:

- $h_t$ = hidden state at time $t$
- $x_t$ = input at time $t$
- $y_t$ = output at time $t$
- $W$ = weight matrices
- $b$ = bias vectors

```python
class VanillaRNN:
    """
    Vanilla RNN implementation from scratch.

    This is the simplest form of RNN to understand the core concept.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        self.hidden_size = hidden_size

        # Initialize weights (small random values)
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        Forward pass through the RNN.

        Args:
            inputs: List of input vectors, each of shape (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)

        Returns:
            outputs: List of output vectors
            hidden_states: List of hidden states
        """
        hidden_states = []
        outputs = []

        h = h_prev

        for x in inputs:
            # Update hidden state
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

            # Compute output
            y = self.Why @ h + self.by

            hidden_states.append(h)
            outputs.append(y)

        return outputs, hidden_states

    def step(self, x, h_prev):
        """
        Single time step.

        Args:
            x: Input at current time step (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)

        Returns:
            y: Output (output_size, 1)
            h: New hidden state (hidden_size, 1)
        """
        h = np.tanh(self.Wxh @ x + self.Whh @ h_prev + self.bh)
        y = self.Why @ h + self.by

        return y, h


def demonstrate_vanilla_rnn():
    """
    Demonstrate how a vanilla RNN processes sequences.
    """
    print("\n" + "="*70)
    print("VANILLA RNN DEMONSTRATION")
    print("="*70)

    # Create RNN
    input_size = 3
    hidden_size = 5
    output_size = 2

    rnn = VanillaRNN(input_size, hidden_size, output_size)

    print(f"\n📊 RNN Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")

    # Count parameters
    total_params = (
        rnn.Wxh.size + rnn.Whh.size + rnn.Why.size +
        rnn.bh.size + rnn.by.size
    )
    print(f"  Total parameters: {total_params}")

    # Create sequence
    sequence_length = 4
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]

    print(f"\n📝 Input sequence length: {sequence_length}")

    # Initial hidden state (zeros)
    h0 = np.zeros((hidden_size, 1))

    # Forward pass
    outputs, hidden_states = rnn.forward(inputs, h0)

    print(f"\n🔄 Processing sequence:")
    for t in range(sequence_length):
        print(f"\n  Time step {t+1}:")
        print(f"    Input shape: {inputs[t].shape}")
        print(f"    Hidden state shape: {hidden_states[t].shape}")
        print(f"    Output shape: {outputs[t].shape}")
        print(f"    Hidden state (first 3 values): {hidden_states[t][:3, 0]}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Hidden state evolution
    h_values = np.concatenate(hidden_states, axis=1)  # (hidden_size, seq_len)

    im = axes[0].imshow(h_values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Hidden Unit', fontsize=12)
    axes[0].set_title('Hidden State Evolution', fontsize=13, weight='bold')
    axes[0].set_xticks(range(sequence_length))
    axes[0].set_xticklabels([f't={i+1}' for i in range(sequence_length)])
    plt.colorbar(im, ax=axes[0], label='Activation')

    # Output evolution
    o_values = np.concatenate(outputs, axis=1)  # (output_size, seq_len)

    for i in range(output_size):
        axes[1].plot(range(sequence_length), o_values[i, :],
                    marker='o', linewidth=2, markersize=8, label=f'Output {i+1}')

    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Output Value', fontsize=12)
    axes[1].set_title('Output Evolution', fontsize=13, weight='bold')
    axes[1].set_xticks(range(sequence_length))
    axes[1].set_xticklabels([f't={i+1}' for i in range(sequence_length)])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week4_vanilla_rnn_demo.png', dpi=150)
    plt.show()

    print("\n✓ RNN processes sequence one step at a time!")
    print("✓ Hidden state carries information across time steps")

demonstrate_vanilla_rnn()
```

### 1.2 Understanding Hidden States

```python
def visualize_hidden_state_memory():
    """
    Visualize how hidden states maintain memory.
    """
    print("\n" + "="*70)
    print("HIDDEN STATE AS MEMORY")
    print("="*70)

    # Simple example: Count number of 1s in binary sequence
    rnn = VanillaRNN(input_size=1, hidden_size=10, output_size=1)

    # Binary sequences
    sequences = [
        [1, 0, 1, 1, 0],  # Three 1s
        [0, 0, 1, 0, 0],  # One 1
        [1, 1, 1, 1, 1],  # Five 1s
    ]

    fig, axes = plt.subplots(len(sequences), 2, figsize=(16, 12))

    for seq_idx, seq in enumerate(sequences):
        # Convert to numpy
        inputs = [np.array([[x]], dtype=np.float32) for x in seq]

        # Process sequence
        h0 = np.zeros((rnn.hidden_size, 1))
        outputs, hidden_states = rnn.forward(inputs, h0)

        # Plot input sequence
        axes[seq_idx, 0].bar(range(len(seq)), seq, alpha=0.7, color='steelblue', edgecolor='black', linewidth=2)
        axes[seq_idx, 0].set_ylim([-0.5, 1.5])
        axes[seq_idx, 0].set_xlabel('Time Step', fontsize=11)
        axes[seq_idx, 0].set_ylabel('Input Value', fontsize=11)
        axes[seq_idx, 0].set_title(f'Sequence {seq_idx+1}: Input', fontsize=12, weight='bold')
        axes[seq_idx, 0].set_xticks(range(len(seq)))
        axes[seq_idx, 0].grid(True, alpha=0.3, axis='y')

        # Plot hidden state evolution
        h_values = np.concatenate(hidden_states, axis=1)

        im = axes[seq_idx, 1].imshow(h_values, cmap='viridis', aspect='auto')
        axes[seq_idx, 1].set_xlabel('Time Step', fontsize=11)
        axes[seq_idx, 1].set_ylabel('Hidden Unit', fontsize=11)
        axes[seq_idx, 1].set_title(f'Sequence {seq_idx+1}: Hidden States', fontsize=12, weight='bold')
        axes[seq_idx, 1].set_xticks(range(len(seq)))
        plt.colorbar(im, ax=axes[seq_idx, 1], label='Activation')

    plt.tight_layout()
    plt.savefig('week4_hidden_state_memory.png', dpi=150)
    plt.show()

    print("\n💡 KEY INSIGHT:")
    print("  Hidden state changes based on input history")
    print("  → Different sequences = Different hidden state patterns")
    print("  → Hidden state = 'Memory' of what happened before")

visualize_hidden_state_memory()
```

### 1.3 Backpropagation Through Time (BPTT)

**Challenge:** How do we train RNNs?

**Solution:** Backpropagation Through Time (BPTT)

**Idea:** "Unroll" the RNN across time and apply standard backpropagation

```python
def visualize_bptt():
    """
    Visualize backpropagation through time.
    """
    print("\n" + "="*70)
    print("BACKPROPAGATION THROUGH TIME (BPTT)")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Forward pass
    time_steps = 5
    x_positions = np.linspace(0.15, 0.85, time_steps)

    axes[0].text(0.5, 0.95, 'Forward Pass: Unrolled RNN', ha='center', fontsize=14, weight='bold')

    # Input sequence
    for i, x in enumerate(x_positions):
        # Input
        axes[0].add_patch(plt.Circle((x, 0.75), 0.025, color='lightblue', ec='blue', linewidth=2))
        axes[0].text(x, 0.68, f'$x_{i}$', ha='center', fontsize=11)

        # RNN cell
        axes[0].add_patch(plt.Rectangle((x-0.04, 0.45), 0.08, 0.1,
                                        facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
        axes[0].text(x, 0.5, 'RNN', ha='center', fontsize=9, weight='bold')

        # Hidden state connection (horizontal)
        if i < time_steps - 1:
            axes[0].annotate('', xy=(x_positions[i+1]-0.04, 0.5), xytext=(x+0.04, 0.5),
                           arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
            axes[0].text((x+x_positions[i+1])/2, 0.53, f'$h_{i}$', ha='center', fontsize=9, color='purple')

        # Input to cell (vertical)
        axes[0].annotate('', xy=(x, 0.55), xytext=(x, 0.72),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

        # Output
        axes[0].add_patch(plt.Circle((x, 0.25), 0.025, color='lightyellow', ec='orange', linewidth=2))
        axes[0].text(x, 0.18, f'$y_{i}$', ha='center', fontsize=11)

        # Cell to output (vertical)
        axes[0].annotate('', xy=(x, 0.275), xytext=(x, 0.45),
                       arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')

    # Backward pass
    axes[1].text(0.5, 0.95, 'Backward Pass: Gradients Flow Back Through Time',
                ha='center', fontsize=14, weight='bold')

    for i, x in enumerate(x_positions):
        # Input
        axes[1].add_patch(plt.Circle((x, 0.75), 0.025, color='lightblue', ec='blue', linewidth=2))
        axes[1].text(x, 0.68, f'$x_{i}$', ha='center', fontsize=11)

        # RNN cell
        axes[1].add_patch(plt.Rectangle((x-0.04, 0.45), 0.08, 0.1,
                                        facecolor='lightcoral', edgecolor='darkred', linewidth=2))
        axes[1].text(x, 0.5, 'RNN', ha='center', fontsize=9, weight='bold')

        # Gradient flow (backward - horizontal)
        if i > 0:
            axes[1].annotate('', xy=(x_positions[i-1]+0.04, 0.5), xytext=(x-0.04, 0.5),
                           arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
            axes[1].text((x+x_positions[i-1])/2, 0.53, '$\\frac{\\partial L}{\\partial h}$',
                        ha='center', fontsize=9, color='red')

        # Output
        axes[1].add_patch(plt.Circle((x, 0.25), 0.025, color='lightyellow', ec='orange', linewidth=2))
        axes[1].text(x, 0.18, f'$L_{i}$', ha='center', fontsize=11, color='red')

        # Gradient from output (vertical)
        axes[1].annotate('', xy=(x, 0.45), xytext=(x, 0.275),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))

    axes[1].text(0.5, 0.05, 'Gradients accumulate as they flow backward through time',
                ha='center', fontsize=11, style='italic', color='red')

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('week4_bptt_visualization.png', dpi=150)
    plt.show()

    print("\n📚 BPTT ALGORITHM:")
    print("-" * 70)
    print("1. Forward Pass:")
    print("   • Process sequence left to right")
    print("   • Compute hidden states h₀, h₁, h₂, ..., hₜ")
    print("   • Compute outputs y₁, y₂, ..., yₜ")
    print("   • Compute loss at each time step")
    print()
    print("2. Backward Pass:")
    print("   • Compute gradients right to left")
    print("   • Gradients flow back through time")
    print("   • Accumulate gradients from all time steps")
    print()
    print("3. Update Weights:")
    print("   • Use accumulated gradients")
    print("   • Apply optimizer (SGD, Adam, etc.)")
    print("-" * 70)

    print("\n✓ BPTT = Standard backprop applied to unrolled RNN")

visualize_bptt()
```

### 1.4 The Vanishing Gradient Problem

**Major Issue:** Gradients diminish as they propagate back through time

**Why?**

$$
\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k}^{t-1} \frac{\partial h_{i+1}}{\partial h_i}
$$

If each term < 1, the product → 0 exponentially fast!

```python
def demonstrate_vanishing_gradients():
    """
    Demonstrate the vanishing gradient problem.
    """
    print("\n" + "="*70)
    print("VANISHING GRADIENT PROBLEM")
    print("="*70)

    # Simulate gradient flow
    sequence_lengths = [10, 20, 30, 40, 50]

    # Different gradient magnitudes
    gradient_factors = {
        'Good (0.95)': 0.95,
        'Borderline (0.9)': 0.9,
        'Bad (0.8)': 0.8,
        'Terrible (0.5)': 0.5
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Gradient magnitude over time
    for label, factor in gradient_factors.items():
        gradients = [factor ** t for t in range(max(sequence_lengths))]
        axes[0].plot(gradients, linewidth=2, label=label, marker='o', markersize=4, alpha=0.7)

    axes[0].set_xlabel('Steps Back in Time', fontsize=12)
    axes[0].set_ylabel('Gradient Magnitude', fontsize=12)
    axes[0].set_title('Gradient Decay Over Time', fontsize=13, weight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1e-5, color='red', linestyle='--', linewidth=2, label='Vanishing threshold')

    # Plot 2: Effective learning range
    factors = list(gradient_factors.values())
    # Calculate how many steps until gradient < 1e-5
    effective_ranges = []
    for factor in factors:
        steps = 0
        grad = 1.0
        while grad > 1e-5 and steps < 100:
            grad *= factor
            steps += 1
        effective_ranges.append(steps)

    bars = axes[1].bar(range(len(gradient_factors)), effective_ranges,
                       color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(gradient_factors))),
                       alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_xticks(range(len(gradient_factors)))
    axes[1].set_xticklabels(gradient_factors.keys(), rotation=45, ha='right')
    axes[1].set_ylabel('Effective Learning Range (Steps)', fontsize=12)
    axes[1].set_title('How Far Back Can RNN Learn?', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, effective_ranges):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val} steps',
                    ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('week4_vanishing_gradients.png', dpi=150)
    plt.show()

    print("\n⚠️  PROBLEM:")
    print("  • Gradients multiply at each time step")
    print("  • If gradient < 1, it shrinks exponentially")
    print("  • After ~20 steps, gradients ≈ 0")
    print("  • RNN can't learn long-term dependencies!")

    print("\n📊 EXAMPLE:")
    print("  Input: 'The cat, which was sitting on the mat, was happy'")
    print("  Task: Predict verb ('was')")
    print("  Problem: Subject ('cat') is 8 words away!")
    print("  → Vanilla RNN struggles with this")

    print("\n💡 SOLUTIONS (Coming on Day 2):")
    print("  1. LSTM (Long Short-Term Memory)")
    print("  2. GRU (Gated Recurrent Unit)")
    print("  → These architectures solve vanishing gradients!")

demonstrate_vanishing_gradients()
```

### 1.5 RNN in PyTorch

```python
def demonstrate_pytorch_rnn():
    """
    Demonstrate PyTorch's built-in RNN.
    """
    print("\n" + "="*70)
    print("PyTorch RNN IMPLEMENTATION")
    print("="*70)

    # Configuration
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_length = 5

    # Create RNN
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True  # Input shape: (batch, seq, features)
    )

    print(f"\n📊 RNN Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in rnn.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create input
    x = torch.randn(batch_size, seq_length, input_size)

    # Initial hidden state (optional, defaults to zeros)
    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    print(f"\n📝 Input:")
    print(f"  Shape: {x.shape} (batch, seq_len, input_size)")
    print(f"  Initial hidden: {h0.shape} (num_layers, batch, hidden_size)")

    # Forward pass
    output, hn = rnn(x, h0)

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape} (batch, seq_len, hidden_size)")
    print(f"  Final hidden: {hn.shape} (num_layers, batch, hidden_size)")

    print("\n💡 KEY POINTS:")
    print("  • output: Hidden states at all time steps")
    print("  • hn: Final hidden state (can be used for next sequence)")
    print("  • PyTorch handles BPTT automatically!")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Input sequence (first sample)
    im1 = axes[0].imshow(x[0].detach().numpy().T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Input Feature', fontsize=12)
    axes[0].set_title('Input Sequence (Sample 1)', fontsize=13, weight='bold')
    axes[0].set_xticks(range(seq_length))
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Output sequence (first sample)
    im2 = axes[1].imshow(output[0].detach().numpy().T, cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Hidden Unit', fontsize=12)
    axes[1].set_title('Hidden States (Sample 1)', fontsize=13, weight='bold')
    axes[1].set_xticks(range(seq_length))
    plt.colorbar(im2, ax=axes[1], label='Activation')

    plt.tight_layout()
    plt.savefig('week4_pytorch_rnn.png', dpi=150)
    plt.show()

    print("\n✓ PyTorch RNN is easy to use and efficient!")

demonstrate_pytorch_rnn()
```

### 1.6 Simple Character-Level RNN

**Real Example:** Train RNN to predict next character

```python
class CharRNN(nn.Module):
    """
    Simple character-level RNN for text generation.
    """

    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer (character → vector)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # RNN layer
        self.rnn = nn.RNN(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input indices [batch, seq_len]
            hidden: Previous hidden state [num_layers, batch, hidden_size]

        Returns:
            output: Predictions [batch, seq_len, vocab_size]
            hidden: New hidden state
        """
        # Embed characters
        embedded = self.embedding(x)  # [batch, seq_len, hidden_size]

        # RNN forward
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)

        # Output predictions
        output = self.fc(output)  # [batch, seq_len, vocab_size]

        return output, hidden


def train_char_rnn_simple():
    """
    Train a simple character RNN on a tiny dataset.
    """
    print("\n" + "="*70)
    print("CHARACTER-LEVEL RNN TRAINING")
    print("="*70)

    # Tiny dataset
    text = "hello hello hello world world world"

    # Create character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"\n📚 Dataset:")
    print(f"  Text: '{text}'")
    print(f"  Length: {len(text)} characters")
    print(f"  Vocabulary: {chars}")
    print(f"  Vocab size: {vocab_size}")

    # Encode text
    encoded = [char_to_idx[ch] for ch in text]

    # Create sequences (input → target)
    seq_length = 5
    sequences = []
    targets = []

    for i in range(len(encoded) - seq_length):
        sequences.append(encoded[i:i+seq_length])
        targets.append(encoded[i+1:i+seq_length+1])

    print(f"\n📝 Training samples: {len(sequences)}")
    print(f"  Example:")
    print(f"    Input: '{text[:seq_length]}' → {sequences[0]}")
    print(f"    Target: '{text[1:seq_length+1]}' → {targets[0]}")

    # Convert to tensors
    X = torch.tensor(sequences, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    # Create model
    hidden_size = 32
    model = CharRNN(vocab_size, hidden_size, num_layers=1)

    print(f"\n🏗️  Model: CharRNN")
    print(f("  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 500
    losses = []

    print(f"\n🏋️  Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()

        # Forward
        output, _ = model(X)

        # Reshape for loss computation
        output = output.reshape(-1, vocab_size)
        targets_flat = Y.reshape(-1)

        loss = criterion(output, targets_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Plot training
    plt.figure(figsize=(12, 5))
    plt.plot(losses, linewidth=2, color='steelblue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Character RNN Training Loss', fontsize=13, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('week4_char_rnn_training.png', dpi=150)
    plt.show()

    print(f"\n✓ Training complete! Final loss: {losses[-1]:.4f}")

    # Generate text
    print("\n📝 GENERATING TEXT:")
    model.eval()

    def generate(model, start_char, length=20):
        """Generate text starting from a character."""
        with torch.no_grad():
            # Encode start character
            current = torch.tensor([[char_to_idx[start_char]]], dtype=torch.long)
            result = start_char
            hidden = None

            for _ in range(length):
                # Predict next character
                output, hidden = model(current, hidden)

                # Sample from distribution
                probs = F.softmax(output[0, -1], dim=0)
                next_idx = torch.multinomial(probs, 1).item()

                next_char = idx_to_char[next_idx]
                result += next_char

                # Update input
                current = torch.tensor([[next_idx]], dtype=torch.long)

            return result

    for start in ['h', 'w', 'l']:
        generated = generate(model, start, length=30)
        print(f"  Starting with '{start}': {generated}")

    print("\n✓ RNN learned to generate character sequences!")

    return model, char_to_idx, idx_to_char

model, char_to_idx, idx_to_char = train_char_rnn_simple()
```

### 1.7 Key Takeaways from Day 1

✅ **RNN Basics**

- Processes sequences one element at a time
- Maintains hidden state (memory)
- Can handle variable-length inputs

✅ **Mathematical Formula**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

- Hidden state updated recursively
- Combines previous hidden state + current input

✅ **Backpropagation Through Time (BPTT)**

- Unroll RNN across time
- Apply standard backpropagation
- Gradients flow back through time

✅ **Vanishing Gradient Problem**

- Gradients diminish exponentially over time
- RNNs struggle with long-term dependencies
- Solution: LSTM and GRU (tomorrow!)

✅ **PyTorch Implementation**

- `nn.RNN`: Built-in RNN layer
- `batch_first=True`: (batch, seq, features)
- Returns output and hidden state

**Tomorrow:** LSTM and GRU - Solving the vanishing gradient problem!

---

_End of Day 1. Total time: 6-8 hours._

---

<a name="day-2"></a>

## 📅 Day 2: LSTM and GRU

> "The only way to do great work is to love what you do." - Steve Jobs

### 2.1 The Problem: Long-Term Dependencies

**Challenge from Day 1:** Vanilla RNNs can't learn long-term dependencies

**Example:**

```
"I grew up in France... [50 words later]... I speak fluent ____"
Answer: "French"
```

Vanilla RNN forgets "France" after 50 words!

```python
def demonstrate_long_term_dependency_problem():
    """
    Show why vanilla RNNs fail at long-term dependencies.
    """
    print("="*70)
    print("LONG-TERM DEPENDENCY PROBLEM")
    print("="*70)

    # Simulate information retention over time
    time_steps = 100

    # Different memory mechanisms
    memories = {
        'Vanilla RNN': [0.9 ** t for t in range(time_steps)],
        'LSTM': [1.0 - 0.005 * t if t < 200 else 0.0 for t in range(time_steps)],
        'Perfect Memory': [1.0] * time_steps
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Information retention
    for name, retention in memories.items():
        linestyle = '-' if name != 'Perfect Memory' else '--'
        linewidth = 3 if name == 'LSTM' else 2
        axes[0].plot(retention, label=name, linewidth=linewidth, linestyle=linestyle, alpha=0.8)

    axes[0].axvline(x=50, color='red', linestyle=':', linewidth=2, alpha=0.5, label='50 steps')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Information Retained', fontsize=12)
    axes[0].set_title('Memory Retention Over Time', fontsize=13, weight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, 1.1])

    # Comparison at key time steps
    steps_to_check = [10, 25, 50, 75, 100]
    vanilla_retention = [memories['Vanilla RNN'][t] for t in steps_to_check]
    lstm_retention = [memories['LSTM'][t] for t in steps_to_check]

    x = np.arange(len(steps_to_check))
    width = 0.35

    bars1 = axes[1].bar(x - width/2, vanilla_retention, width, label='Vanilla RNN',
                       color='coral', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = axes[1].bar(x + width/2, lstm_retention, width, label='LSTM',
                       color='lightgreen', alpha=0.7, edgecolor='black', linewidth=2)

    axes[1].set_xlabel('Time Steps Back', fontsize=12)
    axes[1].set_ylabel('Information Retained', fontsize=12)
    axes[1].set_title('Memory Comparison at Key Steps', fontsize=13, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{t}' for t in steps_to_check])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.1])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('week4_long_term_dependency.png', dpi=150)
    plt.show()

    print("\n📊 PROBLEM:")
    print("  Vanilla RNN after 50 steps: ~0.5% information retained")
    print("  LSTM after 50 steps: ~75% information retained")
    print("  → LSTM maintains memory much longer!")

    print("\n💡 WHY LSTM WORKS:")
    print("  • Cell state: Separate memory pathway")
    print("  • Gates: Control information flow")
    print("  • Constant error carousel: Gradients flow easily")

demonstrate_long_term_dependency_problem()
```

### 2.2 LSTM Architecture

**Long Short-Term Memory (LSTM)** - Hochreiter & Schmidhuber, 1997

**Key Innovation:** **Gates** that control information flow

**Three Gates:**

1. **Forget Gate**: What to forget from cell state
2. **Input Gate**: What new information to store
3. **Output Gate**: What to output

**Mathematical Formulas:**

$$
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate values)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell state update)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)} \\
h_t &= o_t \odot \tanh(C_t) \quad \text{(Hidden state)}
\end{align}
$$

Where:

- $\sigma$ = sigmoid function (outputs 0 to 1)
- $\odot$ = element-wise multiplication
- $C_t$ = cell state (long-term memory)
- $h_t$ = hidden state (short-term memory)

```python
def visualize_lstm_architecture():
    """
    Comprehensive LSTM architecture visualization.
    """
    print("\n" + "="*70)
    print("LSTM ARCHITECTURE")
    print("="*70)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main LSTM cell diagram
    ax_main = fig.add_subplot(gs[:2, :])
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 10)
    ax_main.axis('off')
    ax_main.set_title('LSTM Cell: Complete Architecture', fontsize=16, weight='bold', pad=20)

    # Cell state (top line)
    ax_main.plot([1, 9], [8, 8], 'k-', linewidth=4, alpha=0.3)
    ax_main.text(0.5, 8, '$C_{t-1}$', fontsize=12, weight='bold', ha='right', va='center')
    ax_main.text(9.5, 8, '$C_t$', fontsize=12, weight='bold', ha='left', va='center')
    ax_main.text(5, 8.5, 'Cell State (Long-term Memory)', fontsize=11, ha='center', style='italic')

    # Forget gate (position 2.5)
    forget_x = 2.5
    ax_main.add_patch(plt.Rectangle((forget_x-0.3, 6.5), 0.6, 1, facecolor='#FF6B6B', edgecolor='black', linewidth=2))
    ax_main.text(forget_x, 7, '$f_t$', fontsize=11, ha='center', va='center', weight='bold')
    ax_main.text(forget_x, 6, '$\\sigma$', fontsize=10, ha='center', va='top', style='italic')
    ax_main.arrow(forget_x, 7.5, 0, 0.3, head_width=0.15, head_length=0.1, fc='#FF6B6B', ec='#FF6B6B', linewidth=2)
    ax_main.add_patch(plt.Circle((forget_x, 8), 0.15, facecolor='white', edgecolor='black', linewidth=2))
    ax_main.text(forget_x-0.1, 8, '×', fontsize=14, ha='center', va='center', weight='bold')
    ax_main.text(forget_x, 5.5, 'Forget\nGate', fontsize=9, ha='center', weight='bold', color='#FF6B6B')

    # Input gate (position 5)
    input_x = 5
    ax_main.add_patch(plt.Rectangle((input_x-0.3, 6.5), 0.6, 1, facecolor='#4ECDC4', edgecolor='black', linewidth=2))
    ax_main.text(input_x, 7, '$i_t$', fontsize=11, ha='center', va='center', weight='bold')
    ax_main.text(input_x, 6, '$\\sigma$', fontsize=10, ha='center', va='top', style='italic')

    # Candidate values
    ax_main.add_patch(plt.Rectangle((input_x+0.8, 6.5), 0.6, 1, facecolor='#FFE66D', edgecolor='black', linewidth=2))
    ax_main.text(input_x+1.1, 7, '$\\tilde{C}_t$', fontsize=11, ha='center', va='center', weight='bold')
    ax_main.text(input_x+1.1, 6, 'tanh', fontsize=9, ha='center', va='top', style='italic')

    # Combine input gate and candidate
    ax_main.add_patch(plt.Circle((input_x+0.9, 8), 0.15, facecolor='white', edgecolor='black', linewidth=2))
    ax_main.text(input_x+0.9, 8, '×', fontsize=14, ha='center', va='center', weight='bold')
    ax_main.arrow(input_x, 7.5, 0, 0.3, head_width=0.15, head_length=0.1, fc='#4ECDC4', ec='#4ECDC4', linewidth=2)
    ax_main.arrow(input_x+1.1, 7.5, -0.1, 0.3, head_width=0.15, head_length=0.1, fc='#FFE66D', ec='#FFE66D', linewidth=2)

    ax_main.text(input_x, 5.5, 'Input\nGate', fontsize=9, ha='center', weight='bold', color='#4ECDC4')
    ax_main.text(input_x+1.1, 5.5, 'Candidate\nValues', fontsize=9, ha='center', weight='bold', color='#FFE66D')

    # Add to cell state
    ax_main.add_patch(plt.Circle((6.5, 8), 0.15, facecolor='white', edgecolor='black', linewidth=2))
    ax_main.text(6.5, 8, '+', fontsize=14, ha='center', va='center', weight='bold')

    # Output gate (position 7.5)
    output_x = 7.5
    ax_main.add_patch(plt.Rectangle((output_x-0.3, 6.5), 0.6, 1, facecolor='#95E1D3', edgecolor='black', linewidth=2))
    ax_main.text(output_x, 7, '$o_t$', fontsize=11, ha='center', va='center', weight='bold')
    ax_main.text(output_x, 6, '$\\sigma$', fontsize=10, ha='center', va='top', style='italic')
    ax_main.text(output_x, 5.5, 'Output\nGate', fontsize=9, ha='center', weight='bold', color='#95E1D3')

    # Cell state to output
    ax_main.arrow(8, 8, 0, -1.5, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=2, linestyle='--')
    ax_main.add_patch(plt.Rectangle((output_x+0.8, 6), 0.4, 0.5, facecolor='lightyellow', edgecolor='black', linewidth=1))
    ax_main.text(output_x+1, 6.25, 'tanh', fontsize=9, ha='center', va='center', style='italic')

    # Multiply output gate with tanh(C_t)
    ax_main.add_patch(plt.Circle((output_x, 5), 0.15, facecolor='white', edgecolor='black', linewidth=2))
    ax_main.text(output_x, 5, '×', fontsize=14, ha='center', va='center', weight='bold')
    ax_main.arrow(output_x, 6.5, 0, -1.3, head_width=0.15, head_length=0.1, fc='#95E1D3', ec='#95E1D3', linewidth=2)
    ax_main.arrow(output_x+1, 6, 0, -0.8, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=2)

    # Hidden state output
    ax_main.arrow(output_x, 4.85, 0, -1.5, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax_main.text(output_x, 3, '$h_t$', fontsize=13, ha='center', weight='bold', color='green')
    ax_main.text(output_x, 2.5, 'Hidden State\n(Output)', fontsize=10, ha='center', style='italic', color='green')

    # Inputs
    ax_main.arrow(2, 4, 0, 2.3, head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax_main.text(2, 3.5, '$x_t$', fontsize=12, ha='center', weight='bold', color='blue')
    ax_main.text(2, 3, 'Input', fontsize=10, ha='center', style='italic', color='blue')

    ax_main.arrow(1, 2, 0, 4.3, head_width=0.15, head_length=0.1, fc='purple', ec='purple', linewidth=2, linestyle=':')
    ax_main.text(1, 1.5, '$h_{t-1}$', fontsize=12, ha='center', weight='bold', color='purple')
    ax_main.text(1, 1, 'Previous\nHidden', fontsize=9, ha='center', style='italic', color='purple')

    # Gate equations
    ax_formula = fig.add_subplot(gs[2, 0])
    ax_formula.axis('off')
    ax_formula.text(0.5, 0.9, 'Gate Equations', fontsize=13, weight='bold', ha='center', transform=ax_formula.transAxes)

    equations = [
        ('Forget:', '$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$', '#FF6B6B'),
        ('Input:', '$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$', '#4ECDC4'),
        ('Candidate:', '$\\tilde{C}_t = \\tanh(W_C [h_{t-1}, x_t] + b_C)$', '#FFE66D'),
        ('Cell:', '$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$', 'black'),
        ('Output:', '$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$', '#95E1D3'),
        ('Hidden:', '$h_t = o_t \\odot \\tanh(C_t)$', 'green')
    ]

    y = 0.75
    for label, eq, color in equations:
        ax_formula.text(0.1, y, label, fontsize=10, weight='bold', color=color, transform=ax_formula.transAxes)
        ax_formula.text(0.3, y, eq, fontsize=10, transform=ax_formula.transAxes)
        y -= 0.13

    # Key insights
    ax_insights = fig.add_subplot(gs[2, 1])
    ax_insights.axis('off')
    ax_insights.text(0.5, 0.9, 'Key Insights', fontsize=13, weight='bold', ha='center', transform=ax_insights.transAxes)

    insights = [
        '• Forget gate: Removes irrelevant information',
        '• Input gate: Adds new relevant information',
        '• Cell state: Long-term memory highway',
        '• Output gate: Filters what to output',
        '• Gates use sigmoid → values [0, 1]',
        '• 0 = block all, 1 = pass all'
    ]

    y = 0.75
    for insight in insights:
        ax_insights.text(0.1, y, insight, fontsize=10, transform=ax_insights.transAxes)
        y -= 0.12

    plt.savefig('week4_lstm_architecture.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n🔑 LSTM KEY COMPONENTS:")
    print("-" * 70)
    print("1. Cell State (C_t):")
    print("   • Long-term memory")
    print("   • Flows through entire sequence")
    print("   • Modified by gates")
    print()
    print("2. Hidden State (h_t):")
    print("   • Short-term memory")
    print("   • Output at each time step")
    print("   • Filtered by output gate")
    print()
    print("3. Forget Gate (f_t):")
    print("   • Decides what to forget from C_{t-1}")
    print("   • sigmoid(W_f * [h_{t-1}, x_t])")
    print("   • Output: 0 (forget all) to 1 (remember all)")
    print()
    print("4. Input Gate (i_t) + Candidate (C̃_t):")
    print("   • Decides what new information to add")
    print("   • i_t: How much of candidate to add")
    print("   • C̃_t: New candidate values")
    print()
    print("5. Output Gate (o_t):")
    print("   • Decides what to output")
    print("   • Filters cell state")
    print("-" * 70)

    print("\n✓ LSTM solves vanishing gradients with cell state highway!")

visualize_lstm_architecture()
```

### 2.3 LSTM Implementation from Scratch

```python
class LSTMCell:
    """
    LSTM cell implementation from scratch.

    For educational purposes - shows exactly how LSTM works.
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for gates
        # All gates take [h_{t-1}, x_t] as input
        combined_size = hidden_size + input_size

        # Forget gate
        self.Wf = np.random.randn(hidden_size, combined_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))

        # Input gate
        self.Wi = np.random.randn(hidden_size, combined_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))

        # Candidate values
        self.Wc = np.random.randn(hidden_size, combined_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

        # Output gate
        self.Wo = np.random.randn(hidden_size, combined_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, C_prev):
        """
        LSTM forward pass for one time step.

        Args:
            x: Input (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            C_prev: Previous cell state (hidden_size, 1)

        Returns:
            h: New hidden state
            C: New cell state
            cache: Intermediate values for visualization
        """
        # Concatenate h_prev and x
        combined = np.vstack((h_prev, x))

        # Forget gate
        f = self.sigmoid(self.Wf @ combined + self.bf)

        # Input gate
        i = self.sigmoid(self.Wi @ combined + self.bi)

        # Candidate values
        C_tilde = np.tanh(self.Wc @ combined + self.bc)

        # Update cell state
        C = f * C_prev + i * C_tilde

        # Output gate
        o = self.sigmoid(self.Wo @ combined + self.bo)

        # New hidden state
        h = o * np.tanh(C)

        # Cache for visualization
        cache = {
            'forget_gate': f,
            'input_gate': i,
            'candidate': C_tilde,
            'output_gate': o,
            'cell_state': C
        }

        return h, C, cache


def demonstrate_lstm_forward():
    """
    Demonstrate LSTM forward pass step by step.
    """
    print("\n" + "="*70)
    print("LSTM FORWARD PASS DEMONSTRATION")
    print("="*70)

    # Create LSTM cell
    input_size = 3
    hidden_size = 5
    lstm = LSTMCell(input_size, hidden_size)

    print(f"\n📊 LSTM Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")

    # Process sequence
    sequence_length = 6
    sequence = [np.random.randn(input_size, 1) for _ in range(sequence_length)]

    # Initialize states
    h = np.zeros((hidden_size, 1))
    C = np.zeros((hidden_size, 1))

    # Store values for visualization
    all_gates = {'forget': [], 'input': [], 'output': []}
    all_cell_states = []
    all_hidden_states = []

    print(f"\n🔄 Processing sequence of length {sequence_length}...")

    for t, x in enumerate(sequence):
        h, C, cache = lstm.forward(x, h, C)

        # Store values
        all_gates['forget'].append(cache['forget_gate'].mean())
        all_gates['input'].append(cache['input_gate'].mean())
        all_gates['output'].append(cache['output_gate'].mean())
        all_cell_states.append(cache['cell_state'].copy())
        all_hidden_states.append(h.copy())

        print(f"\n  Step {t+1}:")
        print(f"    Forget gate (avg): {cache['forget_gate'].mean():.4f}")
        print(f"    Input gate (avg): {cache['input_gate'].mean():.4f}")
        print(f"    Output gate (avg): {cache['output_gate'].mean():.4f}")
        print(f"    Cell state (norm): {np.linalg.norm(C):.4f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    time_steps = range(1, sequence_length + 1)

    # Gate activations
    axes[0, 0].plot(time_steps, all_gates['forget'], 'o-', linewidth=2, markersize=8, label='Forget', color='#FF6B6B')
    axes[0, 0].plot(time_steps, all_gates['input'], 's-', linewidth=2, markersize=8, label='Input', color='#4ECDC4')
    axes[0, 0].plot(time_steps, all_gates['output'], '^-', linewidth=2, markersize=8, label='Output', color='#95E1D3')
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time Step', fontsize=12)
    axes[0, 0].set_ylabel('Gate Activation (Average)', fontsize=12)
    axes[0, 0].set_title('Gate Activations Over Time', fontsize=13, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Cell state evolution
    cell_states = np.concatenate(all_cell_states, axis=1)
    im1 = axes[0, 1].imshow(cell_states, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].set_ylabel('Cell State Unit', fontsize=12)
    axes[0, 1].set_title('Cell State Evolution', fontsize=13, weight='bold')
    axes[0, 1].set_xticks(range(sequence_length))
    axes[0, 1].set_xticklabels(time_steps)
    plt.colorbar(im1, ax=axes[0, 1], label='Value')

    # Hidden state evolution
    hidden_states = np.concatenate(all_hidden_states, axis=1)
    im2 = axes[1, 0].imshow(hidden_states, cmap='viridis', aspect='auto')
    axes[1, 0].set_xlabel('Time Step', fontsize=12)
    axes[1, 0].set_ylabel('Hidden State Unit', fontsize=12)
    axes[1, 0].set_title('Hidden State Evolution', fontsize=13, weight='bold')
    axes[1, 0].set_xticks(range(sequence_length))
    axes[1, 0].set_xticklabels(time_steps)
    plt.colorbar(im2, ax=axes[1, 0], label='Value')

    # Cell state magnitude
    cell_magnitudes = [np.linalg.norm(C) for C in all_cell_states]
    axes[1, 1].plot(time_steps, cell_magnitudes, 'o-', linewidth=3, markersize=10, color='purple')
    axes[1, 1].set_xlabel('Time Step', fontsize=12)
    axes[1, 1].set_ylabel('L2 Norm', fontsize=12)
    axes[1, 1].set_title('Cell State Magnitude', fontsize=13, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week4_lstm_forward_pass.png', dpi=150)
    plt.show()

    print("\n✓ LSTM processes sequence maintaining cell state!")
    print("  • Gates control information flow")
    print("  • Cell state carries long-term memory")
    print("  • Hidden state provides output at each step")

demonstrate_lstm_forward()
```

### 2.4 LSTM in PyTorch

```python
def demonstrate_pytorch_lstm():
    """
    Demonstrate PyTorch's LSTM implementation.
    """
    print("\n" + "="*70)
    print("PyTorch LSTM IMPLEMENTATION")
    print("="*70)

    # Configuration
    input_size = 10
    hidden_size = 20
    num_layers = 2  # Stacked LSTMs
    batch_size = 3
    seq_length = 5

    # Create LSTM
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.2  # Dropout between layers
    )

    print(f"\n📊 LSTM Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Dropout: 0.2")

    # Count parameters
    total_params = sum(p.numel() for p in lstm.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Parameter breakdown
    print(f"\n📝 Parameter Breakdown:")
    for name, param in lstm.named_parameters():
        print(f"  {name}: {param.shape} ({param.numel():,} params)")

    # Create input
    x = torch.randn(batch_size, seq_length, input_size)

    # Initial states (optional)
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    print(f"\n📥 Input:")
    print(f"  X shape: {x.shape} (batch, seq_len, input_size)")
    print(f"  h0 shape: {h0.shape} (num_layers, batch, hidden_size)")
    print(f"  c0 shape: {c0.shape} (num_layers, batch, hidden_size)")

    # Forward pass
    output, (hn, cn) = lstm(x, (h0, c0))

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape} (batch, seq_len, hidden_size)")
    print(f"  hn shape: {hn.shape} (num_layers, batch, hidden_size)")
    print(f"  cn shape: {cn.shape} (num_layers, batch, hidden_size)")

    print("\n💡 KEY DIFFERENCES FROM RNN:")
    print("  • Returns TWO states: (hn, cn)")
    print("  • hn: Hidden state (short-term memory)")
    print("  • cn: Cell state (long-term memory)")
    print("  • Cell state maintains memory across long sequences")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sample_idx = 0

    # Input
    im1 = axes[0].imshow(x[sample_idx].detach().numpy().T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Input Feature', fontsize=12)
    axes[0].set_title('Input Sequence', fontsize=13, weight='bold')
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Output (hidden states)
    im2 = axes[1].imshow(output[sample_idx].detach().numpy().T, cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Hidden Unit', fontsize=12)
    axes[1].set_title('Output (Hidden States)', fontsize=13, weight='bold')
    plt.colorbar(im2, ax=axes[1], label='Activation')

    # Final states comparison
    final_h = hn[:, sample_idx, :].detach().numpy()
    final_c = cn[:, sample_idx, :].detach().numpy()

    im3 = axes[2].imshow(np.vstack([final_h, final_c]), cmap='coolwarm', aspect='auto')
    axes[2].set_xlabel('Hidden Unit', fontsize=12)
    axes[2].set_ylabel('State', fontsize=12)
    axes[2].set_title('Final States (h and c)', fontsize=13, weight='bold')
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['h (layer 1)', 'h (layer 2)', 'c (layer 1)', 'c (layer 2)'])
    plt.colorbar(im3, ax=axes[2], label='Value')

    plt.tight_layout()
    plt.savefig('week4_pytorch_lstm.png', dpi=150)
    plt.show()

    print("\n✓ PyTorch LSTM is powerful and easy to use!")

demonstrate_pytorch_lstm()
```

### 2.5 GRU (Gated Recurrent Unit)

**Simpler Alternative to LSTM** - Cho et al., 2014

**Key Idea:** Combine cell state and hidden state into one

**Only Two Gates:**

1. **Reset Gate**: How much of previous hidden state to forget
2. **Update Gate**: How much to update with new information

**Mathematical Formulas:**

$$
\begin{align}
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(Reset gate)} \\
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(Update gate)} \\
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(Candidate)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(New hidden state)}
\end{align}
$$

```python
def visualize_gru_architecture():
    """
    GRU architecture visualization and comparison with LSTM.
    """
    print("\n" + "="*70)
    print("GRU (GATED RECURRENT UNIT) ARCHITECTURE")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # GRU Cell
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('GRU Cell Architecture', fontsize=14, weight='bold')

    # Reset gate
    reset_x = 3
    ax.add_patch(plt.Rectangle((reset_x-0.4, 6), 0.8, 1, facecolor='#FF6B6B', edgecolor='black', linewidth=2))
    ax.text(reset_x, 6.5, '$r_t$', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(reset_x, 5.5, 'Reset\nGate', fontsize=10, ha='center', weight='bold', color='#FF6B6B')

    # Update gate
    update_x = 7
    ax.add_patch(plt.Rectangle((update_x-0.4, 6), 0.8, 1, facecolor='#4ECDC4', edgecolor='black', linewidth=2))
    ax.text(update_x, 6.5, '$z_t$', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(update_x, 5.5, 'Update\nGate', fontsize=10, ha='center', weight='bold', color='#4ECDC4')

    # Candidate hidden state
    ax.add_patch(plt.Rectangle((4.5, 3), 1, 1, facecolor='#FFE66D', edgecolor='black', linewidth=2))
    ax.text(5, 3.5, '$\\tilde{h}_t$', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(5, 2.3, 'Candidate\nHidden', fontsize=10, ha='center', weight='bold', color='#FFE66D')

    # Previous hidden state
    ax.arrow(1, 8, 1.5, 0, head_width=0.2, head_length=0.2, fc='purple', ec='purple', linewidth=2)
    ax.text(0.5, 8, '$h_{t-1}$', fontsize=12, ha='center', weight='bold', color='purple')

    # Input
    ax.arrow(5, 0.5, 0, 1.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue', linewidth=2)
    ax.text(5, 0, '$x_t$', fontsize=12, ha='center', weight='bold', color='blue')

    # Reset applied to h_{t-1}
    ax.add_patch(plt.Circle((3, 3.5), 0.2, facecolor='white', edgecolor='black', linewidth=2))
    ax.text(3, 3.5, '×', fontsize=14, ha='center', va='center', weight='bold')
    ax.arrow(reset_x, 6, 0, -2.3, head_width=0.15, head_length=0.1, fc='#FF6B6B', ec='#FF6B6B', linewidth=2)

    # Combine to form candidate
    ax.arrow(3, 3.5, 1.3, 0, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=2)

    # Update mechanism
    ax.add_patch(plt.Circle((7, 8.5), 0.2, facecolor='white', edgecolor='black', linewidth=2))
    ax.text(7, 8.5, '+', fontsize=14, ha='center', va='center', weight='bold')

    # (1 - z) * h_{t-1}
    ax.text(6, 8.8, '$(1-z_t) \\odot h_{t-1}$', fontsize=9, ha='center')

    # z * h̃_t
    ax.text(7, 4.5, '$z_t \\odot \\tilde{h}_t$', fontsize=9, ha='center')
    ax.arrow(update_x, 6, 0, -1.3, head_width=0.15, head_length=0.1, fc='#4ECDC4', ec='#4ECDC4', linewidth=2)
    ax.arrow(5, 4, 1.8, 0.3, head_width=0.15, head_length=0.1, fc='#FFE66D', ec='#FFE66D', linewidth=2)

    # Output
    ax.arrow(7, 8.7, 0, 0.8, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax.text(7, 9.7, '$h_t$', fontsize=13, ha='center', weight='bold', color='green')

    # Comparison table
    ax = axes[1]
    ax.axis('off')
    ax.set_title('LSTM vs GRU Comparison', fontsize=14, weight='bold')

    comparison = {
        'Property': ['States', 'Gates', 'Parameters', 'Computation', 'Performance', 'When to Use'],
        'LSTM': [
            '2 (h, c)',
            '3 (forget, input, output)',
            'More (~4× input+hidden)',
            'Slower',
            'Better for long sequences',
            'Long-term dependencies\nComplex patterns'
        ],
        'GRU': [
            '1 (h only)',
            '2 (reset, update)',
            'Fewer (~3× input+hidden)',
            'Faster',
            'Similar on many tasks',
            'Shorter sequences\nFaster training needed'
        ]
    }

    # Create table
    y_start = 0.85
    y_step = 0.13

    # Headers
    ax.text(0.15, y_start, 'Property', fontsize=11, weight='bold', ha='center')
    ax.text(0.45, y_start, 'LSTM', fontsize=11, weight='bold', ha='center', color='#FF6B6B')
    ax.text(0.75, y_start, 'GRU', fontsize=11, weight='bold', ha='center', color='#4ECDC4')

    # Draw header line
    ax.plot([0.05, 0.95], [y_start - 0.03, y_start - 0.03], 'k-', linewidth=2)

    # Rows
    y = y_start - y_step
    for i in range(len(comparison['Property'])):
        ax.text(0.15, y, comparison['Property'][i], fontsize=10, ha='center', va='center')
        ax.text(0.45, y, comparison['LSTM'][i], fontsize=9, ha='center', va='center')
        ax.text(0.75, y, comparison['GRU'][i], fontsize=9, ha='center', va='center')
        y -= y_step

        if y > 0.05:
            ax.plot([0.05, 0.95], [y + 0.04, y + 0.04], 'k-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week4_gru_architecture.png', dpi=150)
    plt.show()

    print("\n🔑 GRU KEY FEATURES:")
    print("-" * 70)
    print("1. Simpler than LSTM:")
    print("   • Only 2 gates (vs 3 in LSTM)")
    print("   • Only 1 state (vs 2 in LSTM)")
    print("   • Fewer parameters")
    print()
    print("2. Reset Gate (r_t):")
    print("   • Controls how much of h_{t-1} to forget")
    print("   • Applied before computing candidate")
    print()
    print("3. Update Gate (z_t):")
    print("   • Controls balance between old and new information")
    print("   • Acts like LSTM's forget + input gates combined")
    print()
    print("4. Linear Interpolation:")
    print("   • h_t = (1-z_t) * h_{t-1} + z_t * h̃_t")
    print("   • Smoothly blends old and new")
    print("-" * 70)

    print("\n✓ GRU: Simpler, faster, often just as good as LSTM!")

visualize_gru_architecture()
```

### 2.6 GRU in PyTorch

```python
def compare_lstm_gru_pytorch():
    """
    Compare LSTM and GRU implementations in PyTorch.
    """
    print("\n" + "="*70)
    print("LSTM vs GRU: PyTorch Comparison")
    print("="*70)

    # Configuration
    input_size = 50
    hidden_size = 128
    num_layers = 2
    batch_size = 32
    seq_length = 100

    # Create models
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    # Count parameters
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())

    print(f"\n📊 Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")

    print(f"\n🔢 Parameters:")
    print(f"  LSTM: {lstm_params:,} parameters")
    print(f"  GRU: {gru_params:,} parameters")
    print(f"  Difference: {lstm_params - gru_params:,} ({(1 - gru_params/lstm_params)*100:.1f}% fewer in GRU)")

    # Create input
    x = torch.randn(batch_size, seq_length, input_size)

    # Benchmark speed
    import time

    # LSTM timing
    lstm.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = lstm(x)
        lstm_time = (time.time() - start) / 10

    # GRU timing
    gru.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = gru(x)
        gru_time = (time.time() - start) / 10

    print(f"\n⏱️  Speed (average over 10 runs):")
    print(f"  LSTM: {lstm_time*1000:.2f} ms")
    print(f"  GRU: {gru_time*1000:.2f} ms")
    print(f"  Speedup: {lstm_time/gru_time:.2f}x faster")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Parameters comparison
    models = ['LSTM', 'GRU']
    params = [lstm_params, gru_params]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = axes[0].bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Number of Parameters', fontsize=12)
    axes[0].set_title('Model Size Comparison', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, params):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,}',
                    ha='center', va='bottom', fontsize=11, weight='bold')

    # Speed comparison
    times = [lstm_time * 1000, gru_time * 1000]  # Convert to ms

    bars = axes[1].bar(models, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Time (milliseconds)', fontsize=12)
    axes[1].set_title('Speed Comparison', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f} ms',
                    ha='center', va='bottom', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig('week4_lstm_vs_gru_comparison.png', dpi=150)
    plt.show()

    print("\n💡 PRACTICAL GUIDELINES:")
    print("-" * 70)
    print("Use LSTM when:")
    print("  • You have very long sequences (>100 steps)")
    print("  • Complex temporal dependencies")
    print("  • You have lots of data")
    print("  • Accuracy is more important than speed")
    print()
    print("Use GRU when:")
    print("  • You need faster training")
    print("  • You have limited data")
    print("  • Sequences are moderately long (<100 steps)")
    print("  • You want simpler model")
    print()
    print("In Practice:")
    print("  • GRU is often the first choice")
    print("  • Try both and compare!")
    print("  • Performance is often similar")
    print("-" * 70)

compare_lstm_gru_pytorch()
```

### 2.7 Building a Text Classifier with LSTM

```python
class LSTMTextClassifier(nn.Module):
    """
    LSTM-based text classifier.

    Architecture:
    1. Embedding layer (words → vectors)
    2. LSTM layer (process sequence)
    3. Fully connected layer (classification)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False  # We'll cover bidirectional tomorrow
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        """
        Args:
            text: [batch_size, seq_len] of word indices

        Returns:
            output: [batch_size, output_dim] logits
        """
        # Embed words
        embedded = self.dropout(self.embedding(text))  # [batch, seq, emb_dim]

        # LSTM forward
        output, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state for classification
        hidden = self.dropout(hidden[-1])  # Last layer: [batch, hidden_dim]

        # Classification
        return self.fc(hidden)


def train_lstm_classifier_simple():
    """
    Train LSTM classifier on synthetic data.
    """
    print("\n" + "="*70)
    print("LSTM TEXT CLASSIFIER TRAINING")
    print("="*70)

    # Synthetic dataset: Classify sentences as positive (1) or negative (0)
    # Positive: contains "good", "great", "excellent"
    # Negative: contains "bad", "terrible", "awful"

    vocab = ['<PAD>', 'the', 'movie', 'is', 'good', 'bad', 'great', 'terrible', 'excellent', 'awful', 'very', 'not']
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Training data
    positive_sentences = [
        "the movie is good",
        "the movie is great",
        "the movie is excellent",
        "the movie is very good",
        "the movie is very great",
    ]

    negative_sentences = [
        "the movie is bad",
        "the movie is terrible",
        "the movie is awful",
        "the movie is very bad",
        "the movie is very terrible",
    ]

    sentences = positive_sentences + negative_sentences
    labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

    # Encode sentences
    max_len = max(len(s.split()) for s in sentences)

    def encode_sentence(sentence, max_len):
        words = sentence.split()
        indices = [word_to_idx.get(w, 0) for w in words]
        # Pad to max_len
        indices += [0] * (max_len - len(indices))
        return indices

    X = torch.tensor([encode_sentence(s, max_len) for s in sentences], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    print(f"\n📚 Dataset:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Training samples: {len(sentences)}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Positive samples: {sum(labels)}")
    print(f"  Negative samples: {len(labels) - sum(labels)}")

    print(f"\n📝 Example encoding:")
    print(f"  Sentence: '{sentences[0]}'")
    print(f"  Encoded: {X[0].tolist()}")
    print(f"  Label: {labels[0]} (positive)")

    # Create model
    embedding_dim = 16
    hidden_dim = 32
    output_dim = 2

    model = LSTMTextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0)

    print(f"\n🏗️  Model: LSTMTextClassifier")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 200
    losses = []
    accuracies = []

    print(f"\n🏋️  Training for {num_epochs} epochs...")

    model.train()
    for epoch in range(num_epochs):
        # Forward
        outputs = model(X)
        loss = criterion(outputs, y)

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(accuracy)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.1f}%")

    # Plot training
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(losses, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(accuracies, linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('week4_lstm_classifier_training.png', dpi=150)
    plt.show()

    print(f"\n✓ Training complete! Final accuracy: {accuracies[-1]*100:.1f}%")

    # Test on new sentences
    print("\n📝 TESTING ON NEW SENTENCES:")

    test_sentences = [
        "the movie is very excellent",
        "the movie is very awful",
        "the movie is not good",  # Tricky: "not good" should be negative
    ]

    model.eval()
    with torch.no_grad():
        for sentence in test_sentences:
            # Encode
            encoded = torch.tensor([encode_sentence(sentence, max_len)], dtype=torch.long)

            # Predict
            output = model(encoded)
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()

            sentiment = "Positive" if predicted_class == 1 else "Negative"
            print(f"\n  Sentence: '{sentence}'")
            print(f"  Prediction: {sentiment} (confidence: {confidence*100:.1f}%)")

    print("\n✓ LSTM learned to classify sentiment from word sequences!")

    return model

model = train_lstm_classifier_simple()
```

### 2.8 Key Takeaways from Day 2

✅ **LSTM Architecture**

- 3 gates: Forget, Input, Output
- 2 states: Cell state (long-term), Hidden state (short-term)
- Cell state = information highway
- Solves vanishing gradient problem

✅ **Mathematical Formulas**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- Forget gate: Removes old information
- Input gate: Adds new information
- Output gate: Filters output

✅ **GRU Architecture**

- Simpler: 2 gates (Reset, Update)
- Only 1 state (hidden state)
- Fewer parameters, faster training
- Often performs similarly to LSTM

✅ **LSTM vs GRU**

- LSTM: Better for long sequences, complex patterns
- GRU: Faster, simpler, good default choice
- In practice: Try both!

✅ **PyTorch Implementation**

- `nn.LSTM`: Returns (output, (hn, cn))
- `nn.GRU`: Returns (output, hn)
- Easy to use, handles BPTT automatically

**Tomorrow:** Bidirectional RNNs, Encoder-Decoder, Seq2Seq models!

---

_End of Day 2. Total time: 6-8 hours._

---

<a name="day-3"></a>

## 📅 Day 3: Advanced RNN Architectures

> "Innovation distinguishes between a leader and a follower." - Steve Jobs

### 3.1 Bidirectional RNNs

**Problem:** Standard RNNs only see past context

**Example:**

```
"The movie was not _____"
```

To predict the blank, we need:

- **Past context**: "The movie was not"
- **Future context**: what comes after

**Solution:** Process sequence in **both directions**!

```python
def visualize_bidirectional_concept():
    """
    Visualize bidirectional RNN concept.
    """
    print("="*70)
    print("BIDIRECTIONAL RNN CONCEPT")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Unidirectional RNN
    axes[0].set_title('Unidirectional RNN: Only Past Context', fontsize=14, weight='bold')
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].axis('off')

    # Input sequence
    words = ['The', 'movie', 'was', 'not', '___', 'good']
    x_positions = np.linspace(1, 9, len(words))

    for i, (x, word) in enumerate(zip(x_positions, words)):
        # Input word
        axes[0].add_patch(plt.Circle((x, 7), 0.25, color='lightblue', ec='blue', linewidth=2))
        axes[0].text(x, 6.3, word, ha='center', fontsize=10, weight='bold')

        # RNN cell
        color = 'lightcoral' if i == 4 else 'lightgreen'
        axes[0].add_patch(plt.Rectangle((x-0.3, 4), 0.6, 0.8, facecolor=color, edgecolor='darkgreen', linewidth=2))
        axes[0].text(x, 4.4, 'RNN', ha='center', fontsize=9, weight='bold')

        # Connection to input
        axes[0].arrow(x, 6.65, 0, -2.4, head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2)

        # Forward connection
        if i < len(words) - 1:
            axes[0].arrow(x+0.3, 4.4, x_positions[i+1]-x-0.6, 0,
                         head_width=0.15, head_length=0.15, fc='purple', ec='purple', linewidth=2)

        # Output
        axes[0].add_patch(plt.Circle((x, 2.5), 0.2, color='lightyellow', ec='orange', linewidth=2))
        axes[0].arrow(x, 4, 0, -1.25, head_width=0.15, head_length=0.15, fc='orange', ec='orange', linewidth=2)

    # Highlight problem
    axes[0].text(x_positions[4], 1.5, '❌ Only sees past context!', ha='center', fontsize=11, color='red', weight='bold')
    axes[0].text(5, 0.5, 'Cannot use "good" to predict blank', ha='center', fontsize=10, style='italic', color='red')

    # Bidirectional RNN
    axes[1].set_title('Bidirectional RNN: Past + Future Context', fontsize=14, weight='bold')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis('off')

    for i, (x, word) in enumerate(zip(x_positions, words)):
        # Input word
        axes[1].add_patch(plt.Circle((x, 7), 0.25, color='lightblue', ec='blue', linewidth=2))
        axes[1].text(x, 6.3, word, ha='center', fontsize=10, weight='bold')

        # Forward RNN cell
        color = 'lightcoral' if i == 4 else 'lightgreen'
        axes[1].add_patch(plt.Rectangle((x-0.3, 4.8), 0.6, 0.6, facecolor=color, edgecolor='darkgreen', linewidth=2))
        axes[1].text(x, 5.1, 'F', ha='center', fontsize=9, weight='bold')

        # Backward RNN cell
        axes[1].add_patch(plt.Rectangle((x-0.3, 3.8), 0.6, 0.6, facecolor=color, edgecolor='darkblue', linewidth=2))
        axes[1].text(x, 4.1, 'B', ha='center', fontsize=9, weight='bold')

        # Connection to input
        axes[1].arrow(x, 6.65, 0, -1.5, head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2)

        # Forward connection
        if i < len(words) - 1:
            axes[1].arrow(x+0.3, 5.1, x_positions[i+1]-x-0.6, 0,
                         head_width=0.12, head_length=0.12, fc='purple', ec='purple', linewidth=2)

        # Backward connection
        if i > 0:
            axes[1].arrow(x-0.3, 4.1, -(x-x_positions[i-1])+0.6, 0,
                         head_width=0.12, head_length=0.12, fc='teal', ec='teal', linewidth=2)

        # Combine forward and backward
        axes[1].add_patch(plt.Circle((x, 2.5), 0.2, color='lightyellow', ec='orange', linewidth=2))
        axes[1].arrow(x, 4.8, 0, -2.15, head_width=0.12, head_length=0.12, fc='green', ec='green', linewidth=1.5)
        axes[1].arrow(x, 3.8, 0, -1.15, head_width=0.12, head_length=0.12, fc='cyan', ec='cyan', linewidth=1.5)
        axes[1].text(x+0.5, 3.3, '+', fontsize=12, weight='bold')

    # Highlight solution
    axes[1].text(x_positions[4], 1.5, '✅ Sees both past AND future!', ha='center', fontsize=11, color='green', weight='bold')
    axes[1].text(5, 0.5, 'Can use "good" to help predict blank', ha='center', fontsize=10, style='italic', color='green')

    plt.tight_layout()
    plt.savefig('week4_bidirectional_concept.png', dpi=150)
    plt.show()

    print("\n🔑 BIDIRECTIONAL RNN:")
    print("-" * 70)
    print("1. Forward RNN:")
    print("   • Processes sequence left → right")
    print("   • Captures past context")
    print("   • h_forward_t = f(h_forward_{t-1}, x_t)")
    print()
    print("2. Backward RNN:")
    print("   • Processes sequence right → left")
    print("   • Captures future context")
    print("   • h_backward_t = f(h_backward_{t+1}, x_t)")
    print()
    print("3. Combine:")
    print("   • Concatenate forward and backward hidden states")
    print("   • h_t = [h_forward_t; h_backward_t]")
    print("   • Double the hidden size!")
    print("-" * 70)

    print("\n💡 WHEN TO USE:")
    print("  ✅ Sentence classification (have full sentence)")
    print("  ✅ Named Entity Recognition (NER)")
    print("  ✅ Part-of-speech tagging")
    print("  ✅ Fill-in-the-blank tasks")
    print()
    print("  ❌ Text generation (no future context available)")
    print("  ❌ Real-time applications (can't wait for future)")

visualize_bidirectional_concept()
```

### 3.2 Bidirectional LSTM in PyTorch

```python
def demonstrate_bidirectional_lstm():
    """
    Demonstrate bidirectional LSTM in PyTorch.
    """
    print("\n" + "="*70)
    print("BIDIRECTIONAL LSTM IN PyTorch")
    print("="*70)

    # Configuration
    input_size = 10
    hidden_size = 20
    num_layers = 2
    batch_size = 3
    seq_length = 5

    # Unidirectional LSTM
    lstm_uni = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

    # Bidirectional LSTM
    lstm_bi = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    print(f"\n📊 Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")

    # Count parameters
    uni_params = sum(p.numel() for p in lstm_uni.parameters())
    bi_params = sum(p.numel() for p in lstm_bi.parameters())

    print(f"\n🔢 Parameters:")
    print(f"  Unidirectional: {uni_params:,}")
    print(f"  Bidirectional: {bi_params:,}")
    print(f"  Ratio: {bi_params/uni_params:.1f}x (roughly 2x)")

    # Create input
    x = torch.randn(batch_size, seq_length, input_size)

    print(f"\n📥 Input shape: {x.shape}")

    # Unidirectional forward
    output_uni, (hn_uni, cn_uni) = lstm_uni(x)

    print(f"\n📤 Unidirectional Output:")
    print(f"  Output shape: {output_uni.shape} (batch, seq, hidden)")
    print(f"  Hidden shape: {hn_uni.shape} (num_layers, batch, hidden)")

    # Bidirectional forward
    output_bi, (hn_bi, cn_bi) = lstm_bi(x)

    print(f"\n📤 Bidirectional Output:")
    print(f"  Output shape: {output_bi.shape} (batch, seq, 2*hidden)")
    print(f"  Hidden shape: {hn_bi.shape} (2*num_layers, batch, hidden)")
    print()
    print(f"  ⚠️  Output is concatenation of forward and backward!")
    print(f"     • First {hidden_size} dims: Forward hidden states")
    print(f"     • Last {hidden_size} dims: Backward hidden states")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Unidirectional output
    im1 = axes[0].imshow(output_uni[0].detach().numpy().T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Hidden Unit', fontsize=12)
    axes[0].set_title('Unidirectional LSTM Output', fontsize=13, weight='bold')
    axes[0].set_xticks(range(seq_length))
    plt.colorbar(im1, ax=axes[0], label='Activation')

    # Bidirectional output
    im2 = axes[1].imshow(output_bi[0].detach().numpy().T, cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Hidden Unit', fontsize=12)
    axes[1].set_title('Bidirectional LSTM Output (2x Hidden Size)', fontsize=13, weight='bold')
    axes[1].set_xticks(range(seq_length))
    axes[1].axhline(y=hidden_size-0.5, color='white', linestyle='--', linewidth=2, label='Forward/Backward Split')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='Activation')

    plt.tight_layout()
    plt.savefig('week4_bidirectional_lstm_output.png', dpi=150)
    plt.show()

    print("\n✓ Bidirectional LSTM: 2x hidden size, 2x parameters, better context!")

demonstrate_bidirectional_lstm()
```

### 3.3 Sequence-to-Sequence (Seq2Seq) Models

**Task:** Transform one sequence into another

**Examples:**

- Machine Translation: English → French
- Summarization: Long text → Short summary
- Question Answering: Question → Answer

**Architecture:** Encoder-Decoder

```python
def visualize_seq2seq_architecture():
    """
    Visualize Seq2Seq encoder-decoder architecture.
    """
    print("\n" + "="*70)
    print("SEQUENCE-TO-SEQUENCE (Seq2Seq) ARCHITECTURE")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Example task
    axes[0].set_title('Seq2Seq Example: Machine Translation', fontsize=14, weight='bold')
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].axis('off')

    # Source sentence
    source = ['I', 'love', 'AI', '<EOS>']
    target = ['<SOS>', 'J\'aime', 'IA', '<EOS>']

    # Encoder
    axes[0].text(0.5, 9, 'ENCODER', fontsize=13, weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    enc_x = np.linspace(1, 4, len(source))
    for i, (x, word) in enumerate(zip(enc_x, source)):
        # Input
        axes[0].text(x, 8, word, ha='center', fontsize=11, weight='bold', color='blue')

        # RNN cell
        axes[0].add_patch(plt.Rectangle((x-0.25, 6), 0.5, 0.8, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
        axes[0].text(x, 6.4, 'LSTM', ha='center', fontsize=9, weight='bold')

        # Connection
        axes[0].arrow(x, 7.8, 0, -1.6, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)

        # Forward connection
        if i < len(source) - 1:
            axes[0].arrow(x+0.25, 6.4, enc_x[i+1]-x-0.5, 0,
                         head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)

    # Context vector
    axes[0].add_patch(plt.Circle((5, 6.4), 0.4, facecolor='gold', edgecolor='orange', linewidth=3))
    axes[0].text(5, 6.4, 'Context', ha='center', fontsize=10, weight='bold')
    axes[0].text(5, 5.5, '(Encoder Final State)', ha='center', fontsize=9, style='italic')

    # Arrow from encoder to context
    axes[0].arrow(4.25, 6.4, 0.35, 0, head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=3)

    # Decoder
    axes[0].text(9.5, 9, 'DECODER', fontsize=13, weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Arrow from context to decoder
    axes[0].arrow(5.4, 6.4, 0.5, -1.5, head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=3)

    dec_x = np.linspace(6, 9, len(target))
    for i, (x, word) in enumerate(zip(dec_x, target)):
        # RNN cell
        axes[0].add_patch(plt.Rectangle((x-0.25, 4), 0.5, 0.8, facecolor='lightcoral', edgecolor='darkred', linewidth=2))
        axes[0].text(x, 4.4, 'LSTM', ha='center', fontsize=9, weight='bold')

        # Context connection (first cell)
        if i == 0:
            axes[0].arrow(5.3, 5.8, x-5.3-0.25, -1.6, head_width=0.1, head_length=0.1,
                         fc='green', ec='green', linewidth=2, linestyle='--')

        # Forward connection
        if i < len(target) - 1:
            axes[0].arrow(x+0.25, 4.4, dec_x[i+1]-x-0.5, 0,
                         head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)

        # Input (previous output)
        if i > 0:
            axes[0].arrow(dec_x[i-1], 2.5, x-dec_x[i-1], 1.3,
                         head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2, linestyle=':')

        # Output
        axes[0].text(x, 2.5, word if i > 0 else '<SOS>', ha='center', fontsize=11, weight='bold', color='red')
        axes[0].arrow(x, 4, 0, -1.3, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)

    # Labels
    axes[0].text(2.5, 10, 'Source: "I love AI"', fontsize=11, ha='center', style='italic', color='blue')
    axes[0].text(7.5, 10, 'Target: "J\'aime IA"', fontsize=11, ha='center', style='italic', color='red')

    # Detailed architecture
    axes[1].set_title('Seq2Seq: Training vs Inference', fontsize=14, weight='bold')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis('off')

    # Training mode
    axes[1].text(2.5, 9.5, 'TRAINING: Teacher Forcing', fontsize=12, weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    axes[1].text(1, 8.5, 'Decoder Input:', fontsize=10, weight='bold')
    axes[1].text(1, 8, '<SOS>, J\'aime, IA', fontsize=10, style='italic')
    axes[1].text(4, 8, '→', fontsize=14, ha='center')
    axes[1].text(4.5, 8.5, 'Decoder Output:', fontsize=10, weight='bold')
    axes[1].text(4.5, 8, 'J\'aime, IA, <EOS>', fontsize=10, style='italic')

    axes[1].text(2.5, 7, '✅ Use ground truth as input', fontsize=10, ha='center', color='green')
    axes[1].text(2.5, 6.5, '✅ Faster training', fontsize=10, ha='center', color='green')

    # Inference mode
    axes[1].text(7.5, 9.5, 'INFERENCE: Auto-regressive', fontsize=12, weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    axes[1].text(6, 8.5, 'Step 1:', fontsize=10, weight='bold')
    axes[1].text(6, 8, 'Input: <SOS> → Output: J\'aime', fontsize=10, style='italic')

    axes[1].text(6, 7.5, 'Step 2:', fontsize=10, weight='bold')
    axes[1].text(6, 7, 'Input: J\'aime → Output: IA', fontsize=10, style='italic')

    axes[1].text(6, 6.5, 'Step 3:', fontsize=10, weight='bold')
    axes[1].text(6, 6, 'Input: IA → Output: <EOS>', fontsize=10, style='italic')

    axes[1].text(7.5, 5, '✅ Use previous prediction as input', fontsize=10, ha='center', color='green')
    axes[1].text(7.5, 4.5, '⚠️  Errors accumulate!', fontsize=10, ha='center', color='orange')

    # Key components
    axes[1].text(5, 3, 'KEY COMPONENTS', fontsize=12, weight='bold', ha='center')
    axes[1].text(1, 2, '• Encoder: Compress source into context vector', fontsize=10)
    axes[1].text(1, 1.5, '• Context: Fixed-size representation of source', fontsize=10)
    axes[1].text(1, 1, '• Decoder: Generate target from context', fontsize=10)
    axes[1].text(1, 0.5, '• <SOS>: Start-of-sequence token', fontsize=10)
    axes[1].text(5.5, 2, '• <EOS>: End-of-sequence token', fontsize=10)
    axes[1].text(5.5, 1.5, '• Teacher Forcing: Use ground truth during training', fontsize=10)
    axes[1].text(5.5, 1, '• Auto-regressive: Use predictions during inference', fontsize=10)

    plt.tight_layout()
    plt.savefig('week4_seq2seq_architecture.png', dpi=150)
    plt.show()

    print("\n🔑 SEQ2SEQ KEY CONCEPTS:")
    print("-" * 70)
    print("1. Encoder:")
    print("   • Processes source sequence")
    print("   • Outputs context vector (final hidden state)")
    print("   • Compresses source information")
    print()
    print("2. Context Vector:")
    print("   • Fixed-size representation of source")
    print("   • Bottleneck: Must contain all source info!")
    print("   • Passed to decoder as initial state")
    print()
    print("3. Decoder:")
    print("   • Generates target sequence")
    print("   • Initialized with context vector")
    print("   • Auto-regressive: Uses previous outputs")
    print()
    print("4. Training (Teacher Forcing):")
    print("   • Use ground truth as decoder input")
    print("   • Faster, more stable training")
    print("   • Input: <SOS>, y1, y2, ...")
    print("   • Output: y1, y2, y3, <EOS>")
    print()
    print("5. Inference:")
    print("   • Auto-regressive generation")
    print("   • Feed previous prediction as input")
    print("   • Stop when <EOS> generated or max length")
    print("-" * 70)

    print("\n💡 LIMITATION:")
    print("  ⚠️  Context vector is a bottleneck!")
    print("  → All source information compressed into fixed-size vector")
    print("  → Long sequences lose information")
    print("  → Solution: ATTENTION MECHANISM (Day 4)!")

visualize_seq2seq_architecture()
```

### 3.4 Implementing Seq2Seq in PyTorch

```python
class Encoder(nn.Module):
    """
    Encoder: Compress source sequence into context vector.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [batch_size, src_len]

        Returns:
            hidden: [n_layers, batch, hid_dim]
            cell: [n_layers, batch, hid_dim]
        """
        # Embed source
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]

        # Encode
        outputs, (hidden, cell) = self.rnn(embedded)

        # Return final states
        return hidden, cell


class Decoder(nn.Module):
    """
    Decoder: Generate target sequence from context.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        Args:
            input: [batch_size] (single time step)
            hidden: [n_layers, batch, hid_dim]
            cell: [n_layers, batch, hid_dim]

        Returns:
            prediction: [batch, output_dim]
            hidden: [n_layers, batch, hid_dim]
            cell: [n_layers, batch, hid_dim]
        """
        # Add sequence dimension
        input = input.unsqueeze(1)  # [batch, 1]

        # Embed
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        # Decode
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Prediction
        prediction = self.fc_out(output.squeeze(1))  # [batch, output_dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Seq2Seq model: Encoder + Decoder.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch, src_len]
            trg: [batch, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: [batch, trg_len, output_dim]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode source
        hidden, cell = self.encoder(src)

        # First input to decoder is <SOS>
        input = trg[:, 0]

        # Decode
        for t in range(1, trg_len):
            # Decode one step
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Store output
            outputs[:, t] = output

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            # Next input
            top1 = output.argmax(1)
            input = trg[:, t] if use_teacher_forcing else top1

        return outputs


def demonstrate_seq2seq():
    """
    Demonstrate Seq2Seq model.
    """
    print("\n" + "="*70)
    print("SEQ2SEQ MODEL DEMONSTRATION")
    print("="*70)

    # Hyperparameters
    INPUT_DIM = 100   # Source vocabulary size
    OUTPUT_DIM = 100  # Target vocabulary size
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    HID_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    print(f"\n🏗️  Seq2Seq Model:")
    print(f"  Source vocab size: {INPUT_DIM}")
    print(f"  Target vocab size: {OUTPUT_DIM}")
    print(f"  Hidden dim: {HID_DIM}")
    print(f"  Num layers: {N_LAYERS}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Breakdown
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"    Encoder: {enc_params:,}")
    print(f"    Decoder: {dec_params:,}")

    # Example forward pass
    batch_size = 4
    src_len = 10
    trg_len = 12

    src = torch.randint(0, INPUT_DIM, (batch_size, src_len)).to(device)
    trg = torch.randint(0, OUTPUT_DIM, (batch_size, trg_len)).to(device)

    print(f"\n📥 Input:")
    print(f"  Source shape: {src.shape} (batch, src_len)")
    print(f"  Target shape: {trg.shape} (batch, trg_len)")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(src, trg, teacher_forcing_ratio=0.5)

    print(f"\n📤 Output:")
    print(f"  Shape: {output.shape} (batch, trg_len, output_dim)")
    print(f"  Contains logits for each target token")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Source sequence (first sample)
    axes[0].imshow(src[0:1].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Source Position', fontsize=12)
    axes[0].set_ylabel('Sample', fontsize=12)
    axes[0].set_title('Source Sequence (Token IDs)', fontsize=13, weight='bold')
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['Sample 1'])

    # Output predictions (first sample, argmax)
    predictions = output.argmax(dim=-1)[0].cpu().numpy()
    axes[1].imshow(predictions.reshape(1, -1), cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Target Position', fontsize=12)
    axes[1].set_ylabel('Sample', fontsize=12)
    axes[1].set_title('Predicted Target Sequence (Token IDs)', fontsize=13, weight='bold')
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['Sample 1'])

    plt.tight_layout()
    plt.savefig('week4_seq2seq_demo.png', dpi=150)
    plt.show()

    print("\n✓ Seq2Seq model successfully encodes and decodes sequences!")
    print("  → Encoder compresses source")
    print("  → Decoder generates target")
    print("  → Teacher forcing during training")

demonstrate_seq2seq()
```

### 3.5 Key Takeaways from Day 3

✅ **Bidirectional RNNs**

- Process sequence in both directions
- Concatenate forward and backward hidden states
- Better context understanding
- 2x hidden size, 2x parameters
- Use when full sequence available

✅ **Seq2Seq Architecture**

- Encoder: Compress source → context vector
- Decoder: Generate target from context
- Two separate RNNs
- Variable input/output lengths

✅ **Training vs Inference**

- Training: Teacher forcing (use ground truth)
- Inference: Auto-regressive (use predictions)
- Teacher forcing ratio: Balance between stability and realism

✅ **Limitations**

- Context vector bottleneck
- Long sequences lose information
- Fixed-size representation problematic
- Solution: Attention mechanisms (Day 4)!

✅ **Applications**

- Machine translation
- Text summarization
- Chatbots
- Image captioning
- Speech recognition

**Tomorrow:** Attention mechanisms - solving the bottleneck problem!

---

_End of Day 3. Total time: 6-8 hours._

---

<a name="day-4"></a>

## 📅 Day 4: Attention Mechanisms

> "Attention is all you need." - Vaswani et al., 2017

### 4.1 The Seq2Seq Bottleneck Problem

**Recap from Day 3:** Seq2Seq models compress entire source sequence into a fixed-size context vector.

**Problem:** Information bottleneck!

```python
def visualize_bottleneck_problem():
    """
    Visualize the context vector bottleneck in Seq2Seq.
    """
    print("="*70)
    print("THE CONTEXT VECTOR BOTTLENECK")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Short sentence
    axes[0].set_title('Short Sentence: Bottleneck Works OK', fontsize=14, weight='bold')
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].axis('off')

    short_source = ['I', 'love', 'AI']
    x_pos = np.linspace(1, 3, len(short_source))

    for i, (x, word) in enumerate(zip(x_pos, short_source)):
        axes[0].add_patch(plt.Circle((x, 7), 0.3, color='lightblue', ec='blue', linewidth=2))
        axes[0].text(x, 6.3, word, ha='center', fontsize=11, weight='bold')

    # Context vector (small)
    axes[0].add_patch(plt.Circle((5, 7), 0.5, color='gold', ec='orange', linewidth=3))
    axes[0].text(5, 7, 'Context\n(3 words)', ha='center', fontsize=10, weight='bold')
    axes[0].arrow(3.3, 7, 1.2, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)

    axes[0].text(5, 5, '✅ All information fits!', ha='center', fontsize=12, color='green', weight='bold')

    # Long sentence
    axes[1].set_title('Long Sentence: Bottleneck Loses Information', fontsize=14, weight='bold')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis('off')

    long_source = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'in', 'garden']
    x_pos_long = np.linspace(0.5, 3.5, len(long_source))

    for i, (x, word) in enumerate(zip(x_pos_long, long_source)):
        axes[1].add_patch(plt.Circle((x, 7), 0.15, color='lightblue', ec='blue', linewidth=1))
        axes[1].text(x, 6, word, ha='center', fontsize=7, rotation=45)

    # Context vector (same size!)
    axes[1].add_patch(plt.Circle((5, 7), 0.5, color='red', ec='darkred', linewidth=3))
    axes[1].text(5, 7, 'Context\n(10 words)', ha='center', fontsize=10, weight='bold')
    axes[1].arrow(3.8, 7, 0.7, 0, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=3)

    axes[1].text(5, 5, '❌ Information lost!', ha='center', fontsize=12, color='red', weight='bold')
    axes[1].text(5, 4.3, 'Same size vector for 10 words', ha='center', fontsize=10, style='italic')

    # Add graph showing performance
    axes[1].text(7.5, 8.5, 'Translation Quality', fontsize=11, weight='bold')
    seq_lengths = [5, 10, 20, 30, 40, 50]
    quality = [0.9, 0.85, 0.75, 0.65, 0.55, 0.45]  # Hypothetical scores

    for i, (length, qual) in enumerate(zip(seq_lengths, quality)):
        bar_height = qual * 3
        axes[1].add_patch(plt.Rectangle((7 + i*0.4, 5), 0.3, bar_height,
                                        facecolor='coral' if qual < 0.7 else 'lightgreen',
                                        edgecolor='black', linewidth=1))
        axes[1].text(7.15 + i*0.4, 4.7, str(length), ha='center', fontsize=8)

    axes[1].text(8.5, 4.3, 'Sequence Length', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('week4_bottleneck_problem.png', dpi=150)
    plt.show()

    print("\n⚠️  BOTTLENECK PROBLEM:")
    print("-" * 70)
    print("• Context vector has FIXED SIZE (e.g., 512 dimensions)")
    print("• Must encode entire source sequence")
    print("• Short sequence (3 words): 512 dims → plenty of space")
    print("• Long sequence (50 words): 512 dims → information loss!")
    print()
    print("📊 EMPIRICAL EVIDENCE:")
    print("• Seq2Seq performance degrades with sequence length")
    print("• Translation quality drops significantly after 20-30 words")
    print("• Critical information from early tokens gets forgotten")
    print("-" * 70)

    print("\n💡 SOLUTION: ATTENTION MECHANISM")
    print("  → Instead of fixed context, let decoder 'attend' to ALL encoder states")
    print("  → Focus on relevant parts of source at each decoding step")
    print("  → No information bottleneck!")

visualize_bottleneck_problem()
```

### 4.2 Attention Mechanism: The Idea

**Core Concept:** At each decoding step, look at ALL encoder hidden states, not just the last one!

**How?** Calculate "attention weights" that determine which encoder states to focus on.

```python
def visualize_attention_concept():
    """
    Visualize the attention mechanism concept.
    """
    print("\n" + "="*70)
    print("ATTENTION MECHANISM CONCEPT")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Without attention
    ax = axes[0]
    ax.set_title('WITHOUT Attention: Fixed Context', fontsize=14, weight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Encoder
    source = ['The', 'cat', 'sat', 'on', 'mat']
    x_enc = np.linspace(1, 5, len(source))

    for i, (x, word) in enumerate(zip(x_enc, source)):
        ax.add_patch(plt.Circle((x, 8), 0.25, color='lightblue', ec='blue', linewidth=2))
        ax.text(x, 7.2, word, ha='center', fontsize=10, weight='bold')
        ax.add_patch(plt.Rectangle((x-0.2, 6), 0.4, 0.5, facecolor='lightgreen', edgecolor='green', linewidth=1))
        ax.text(x, 6.25, 'E', ha='center', fontsize=9, weight='bold')

    # Only last encoder state used
    ax.arrow(5.2, 6.25, 1.3, -0.5, head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=3)

    # Decoder
    target = ['Le', 'chat']
    x_dec = [7, 8.5]

    for i, (x, word) in enumerate(zip(x_dec, target)):
        ax.add_patch(plt.Rectangle((x-0.2, 5), 0.4, 0.5, facecolor='lightcoral', edgecolor='red', linewidth=1))
        ax.text(x, 5.25, 'D', ha='center', fontsize=9, weight='bold')
        ax.text(x, 4.3, word, ha='center', fontsize=10, weight='bold')

    ax.text(7.75, 3, '❌ Only uses final encoder state', ha='center', fontsize=11, color='red', weight='bold')

    # With attention
    ax = axes[1]
    ax.set_title('WITH Attention: Dynamic Context', fontsize=14, weight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Encoder
    for i, (x, word) in enumerate(zip(x_enc, source)):
        ax.add_patch(plt.Circle((x, 8), 0.25, color='lightblue', ec='blue', linewidth=2))
        ax.text(x, 7.2, word, ha='center', fontsize=10, weight='bold')
        ax.add_patch(plt.Rectangle((x-0.2, 6), 0.4, 0.5, facecolor='lightgreen', edgecolor='green', linewidth=1))
        ax.text(x, 6.25, 'E', ha='center', fontsize=9, weight='bold')

    # Decoder attends to all encoder states
    # First decoder step: focus on "The" and "cat"
    attention_weights_1 = [0.4, 0.5, 0.05, 0.03, 0.02]  # Higher on "The" and "cat"

    for i, (x, weight) in enumerate(zip(x_enc, attention_weights_1)):
        # Attention line
        alpha = weight  # Opacity based on weight
        linewidth = 1 + weight * 4
        ax.plot([x, 7], [6, 5.5], 'g-', alpha=alpha, linewidth=linewidth)
        ax.text(x, 5.5, f'{weight:.2f}', ha='center', fontsize=8, color='green', weight='bold')

    ax.add_patch(plt.Rectangle((7-0.2, 5), 0.4, 0.5, facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax.text(7, 5.25, 'D₁', ha='center', fontsize=9, weight='bold')
    ax.text(7, 4.3, 'Le', ha='center', fontsize=10, weight='bold')

    # Second decoder step: focus on "cat" and "sat"
    attention_weights_2 = [0.05, 0.6, 0.3, 0.03, 0.02]  # Higher on "cat" and "sat"

    for i, (x, weight) in enumerate(zip(x_enc, attention_weights_2)):
        alpha = weight
        linewidth = 1 + weight * 4
        ax.plot([x, 8.5], [6, 5.5], 'b-', alpha=alpha, linewidth=linewidth)

    ax.add_patch(plt.Rectangle((8.5-0.2, 5), 0.4, 0.5, facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax.text(8.5, 5.25, 'D₂', ha='center', fontsize=9, weight='bold')
    ax.text(8.5, 4.3, 'chat', ha='center', fontsize=10, weight='bold')

    ax.text(7.75, 3, '✅ Attends to ALL encoder states', ha='center', fontsize=11, color='green', weight='bold')
    ax.text(7.75, 2.5, 'Different focus at each step', ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('week4_attention_concept.png', dpi=150)
    plt.show()

    print("\n🔑 ATTENTION KEY IDEAS:")
    print("-" * 70)
    print("1. Access to All Encoder States:")
    print("   • Don't compress into single vector")
    print("   • Keep all encoder hidden states h₁, h₂, ..., hₙ")
    print()
    print("2. Attention Weights:")
    print("   • At each decoder step, calculate weights α₁, α₂, ..., αₙ")
    print("   • Weights sum to 1: Σαᵢ = 1")
    print("   • High weight = focus on that encoder state")
    print()
    print("3. Context Vector (Dynamic):")
    print("   • c_t = Σ αᵢ * hᵢ")
    print("   • Weighted sum of encoder states")
    print("   • Different context at each decoding step!")
    print()
    print("4. Benefits:")
    print("   • No fixed-size bottleneck")
    print("   • Model learns where to focus")
    print("   • Better performance on long sequences")
    print("   • Interpretable (visualize attention weights)")
    print("-" * 70)

visualize_attention_concept()
```

### 4.3 Attention Mechanism: Mathematics

**Bahdanau Attention** (Additive Attention) - 2014

Given:

- Decoder hidden state at time $t$: $s_t$
- Encoder hidden states: $h_1, h_2, ..., h_n$

**Step 1: Calculate attention scores**
$$e_{ti} = v^T \tanh(W_1 s_t + W_2 h_i)$$

**Step 2: Normalize to get attention weights** (softmax)
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{n} \exp(e_{tj})}$$

**Step 3: Calculate context vector**
$$c_t = \sum_{i=1}^{n} \alpha_{ti} h_i$$

**Step 4: Combine context with decoder state**
$$\tilde{s}_t = \tanh(W_c [c_t; s_t])$$

```python
class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.

    Paper: "Neural Machine Translation by Jointly Learning to Align and Translate"
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Attention parameters
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder state
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder states
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # Attention score

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: [batch, hidden_dim] - current decoder state
            encoder_outputs: [batch, src_len, hidden_dim] - all encoder states

        Returns:
            context: [batch, hidden_dim] - context vector
            attention_weights: [batch, src_len] - attention weights (for visualization)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Expand decoder hidden to match encoder outputs
        # [batch, hidden] → [batch, src_len, hidden]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate attention scores
        # e = v^T * tanh(W1*s + W2*h)
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))
        attention_scores = self.v(energy).squeeze(2)  # [batch, src_len]

        # Normalize with softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, src_len]

        # Calculate context vector
        # c = Σ α_i * h_i
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
        context = context.squeeze(1)  # [batch, hidden]

        return context, attention_weights


def demonstrate_attention():
    """
    Demonstrate attention mechanism.
    """
    print("\n" + "="*70)
    print("BAHDANAU ATTENTION DEMONSTRATION")
    print("="*70)

    # Parameters
    batch_size = 2
    src_len = 6
    hidden_dim = 8

    # Create attention module
    attention = BahdanauAttention(hidden_dim)

    # Create dummy encoder outputs and decoder hidden state
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
    decoder_hidden = torch.randn(batch_size, hidden_dim)

    print(f"\n📊 Input:")
    print(f"  Encoder outputs shape: {encoder_outputs.shape}")
    print(f"  Decoder hidden shape: {decoder_hidden.shape}")

    # Forward pass
    context, attention_weights = attention(decoder_hidden, encoder_outputs)

    print(f"\n📤 Output:")
    print(f"  Context vector shape: {context.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")

    print(f"\n🔍 Attention Weights (Sample 1):")
    weights = attention_weights[0].detach().numpy()
    print(f"  Weights: {weights}")
    print(f"  Sum: {weights.sum():.4f} (should be 1.0)")
    print(f"  Max weight at position: {weights.argmax()}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Attention weights heatmap
    im = axes[0].imshow(attention_weights.detach().numpy(), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xlabel('Source Position', fontsize=12)
    axes[0].set_ylabel('Sample', fontsize=12)
    axes[0].set_title('Attention Weights', fontsize=13, weight='bold')
    axes[0].set_xticks(range(src_len))
    axes[0].set_yticks(range(batch_size))
    plt.colorbar(im, ax=axes[0], label='Weight')

    # Add values
    for i in range(batch_size):
        for j in range(src_len):
            text = axes[0].text(j, i, f'{attention_weights[i, j].item():.2f}',
                               ha="center", va="center", color="black", fontsize=10, weight='bold')

    # Attention weights as bar chart (sample 1)
    axes[1].bar(range(src_len), weights, color='coral', alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Source Position', fontsize=12)
    axes[1].set_ylabel('Attention Weight', fontsize=12)
    axes[1].set_title('Attention Distribution (Sample 1)', fontsize=13, weight='bold')
    axes[1].set_xticks(range(src_len))
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Highlight max
    max_idx = weights.argmax()
    axes[1].bar(max_idx, weights[max_idx], color='red', alpha=0.9, edgecolor='darkred', linewidth=3)
    axes[1].text(max_idx, weights[max_idx] + 0.05, 'Focus here!',
                ha='center', fontsize=10, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('week4_attention_demo.png', dpi=150)
    plt.show()

    print("\n✓ Attention mechanism successfully computed!")
    print("  → Weights sum to 1.0")
    print("  → Higher weights = more focus")
    print("  → Context = weighted sum of encoder states")

demonstrate_attention()
```

### 4.4 Luong Attention (Multiplicative Attention)

**Alternative:** Simpler dot-product attention - 2015

**Three variants:**

1. **Dot:** $\text{score}(s_t, h_i) = s_t^T h_i$
2. **General:** $\text{score}(s_t, h_i) = s_t^T W h_i$
3. **Concat:** $\text{score}(s_t, h_i) = v^T \tanh(W[s_t; h_i])$ (similar to Bahdanau)

```python
class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention.

    Paper: "Effective Approaches to Attention-based Neural Machine Translation"
    """

    def __init__(self, hidden_dim, method='general'):
        super().__init__()

        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: [batch, hidden_dim]
            encoder_outputs: [batch, src_len, hidden_dim]

        Returns:
            context: [batch, hidden_dim]
            attention_weights: [batch, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        if self.method == 'dot':
            # score(s, h) = s^T h
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))  # [batch, src_len, 1]
            attention_scores = attention_scores.squeeze(2)  # [batch, src_len]

        elif self.method == 'general':
            # score(s, h) = s^T W h
            # Transform encoder outputs
            transformed = self.W(encoder_outputs)  # [batch, src_len, hidden]
            attention_scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2))  # [batch, src_len, 1]
            attention_scores = attention_scores.squeeze(2)  # [batch, src_len]

        elif self.method == 'concat':
            # score(s, h) = v^T tanh(W [s; h])
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            concatenated = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)  # [batch, src_len, 2*hidden]
            energy = torch.tanh(self.W(concatenated))  # [batch, src_len, hidden]
            attention_scores = self.v(energy).squeeze(2)  # [batch, src_len]

        # Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=1)

        # Context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


def compare_attention_mechanisms():
    """
    Compare Bahdanau and Luong attention mechanisms.
    """
    print("\n" + "="*70)
    print("ATTENTION MECHANISMS COMPARISON")
    print("="*70)

    # Parameters
    batch_size = 1
    src_len = 8
    hidden_dim = 16

    # Create different attention modules
    bahdanau = BahdanauAttention(hidden_dim)
    luong_dot = LuongAttention(hidden_dim, method='dot')
    luong_general = LuongAttention(hidden_dim, method='general')
    luong_concat = LuongAttention(hidden_dim, method='concat')

    # Dummy inputs
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
    decoder_hidden = torch.randn(batch_size, hidden_dim)

    # Get attention weights from each
    _, attn_bahdanau = bahdanau(decoder_hidden, encoder_outputs)
    _, attn_dot = luong_dot(decoder_hidden, encoder_outputs)
    _, attn_general = luong_general(decoder_hidden, encoder_outputs)
    _, attn_concat = luong_concat(decoder_hidden, encoder_outputs)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    mechanisms = [
        ('Bahdanau (Additive)', attn_bahdanau, axes[0, 0]),
        ('Luong Dot', attn_dot, axes[0, 1]),
        ('Luong General', attn_general, axes[1, 0]),
        ('Luong Concat', attn_concat, axes[1, 1])
    ]

    for name, weights, ax in mechanisms:
        weights_np = weights[0].detach().numpy()

        bars = ax.bar(range(src_len), weights_np, alpha=0.7, edgecolor='black', linewidth=2)

        # Color bars by weight
        colors = plt.cm.YlOrRd(weights_np / weights_np.max())
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

        ax.set_xlabel('Source Position', fontsize=11)
        ax.set_ylabel('Attention Weight', fontsize=11)
        ax.set_title(name, fontsize=12, weight='bold')
        ax.set_ylim([0, max(weights_np.max() * 1.2, 0.3)])
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight max
        max_idx = weights_np.argmax()
        ax.text(max_idx, weights_np[max_idx] + 0.02, f'{weights_np[max_idx]:.3f}',
               ha='center', fontsize=9, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('week4_attention_comparison.png', dpi=150)
    plt.show()

    print("\n📊 ATTENTION MECHANISMS COMPARED:")
    print("-" * 70)
    print("| Mechanism      | Complexity | Parameters | Speed    | Performance |")
    print("|----------------|------------|------------|----------|-------------|")
    print("| Bahdanau       | Medium     | 3W + v     | Medium   | Very Good   |")
    print("| Luong Dot      | Low        | 0          | Fastest  | Good        |")
    print("| Luong General  | Low        | W          | Fast     | Very Good   |")
    print("| Luong Concat   | Medium     | W + v      | Medium   | Very Good   |")
    print("-" * 70)

    print("\n💡 WHICH TO USE?")
    print("  • Luong Dot: Fastest, no parameters, requires same hidden sizes")
    print("  • Luong General: Good balance, works with different sizes")
    print("  • Bahdanau: Original, proven effective")
    print("  • Luong Concat: Similar to Bahdanau, flexible")
    print()
    print("  → In practice: Try multiple, see what works best!")
    print("  → Modern Transformers use Scaled Dot-Product (Day 5)")

compare_attention_mechanisms()
```

### 4.5 Seq2Seq with Attention

```python
class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism.
    """

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # RNN input = embedding + context vector
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim)

        # Output layer: combines decoder state, context, and embedding
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        Args:
            input: [batch_size] - current target token
            hidden: [1, batch, dec_hid_dim] - previous decoder hidden
            encoder_outputs: [batch, src_len, enc_hid_dim] - all encoder outputs

        Returns:
            prediction: [batch, output_dim]
            hidden: [1, batch, dec_hid_dim]
            attention_weights: [batch, src_len]
        """
        # Embed input
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]

        # Calculate attention
        context, attention_weights = self.attention(hidden.squeeze(0), encoder_outputs)

        # Concatenate embedded input and context
        rnn_input = torch.cat([embedded, context.unsqueeze(0)], dim=2)  # [1, batch, emb+enc_hid]

        # RNN forward
        output, hidden = self.rnn(rnn_input, hidden)

        # Prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context

        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))

        return prediction, hidden, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Seq2Seq model with attention mechanism.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch, src_len]
            trg: [batch, trg_len]

        Returns:
            outputs: [batch, trg_len, output_dim]
            attentions: [batch, trg_len, src_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src)

        # First input
        input = trg[:, 0]

        # Decode
        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input, hidden.unsqueeze(0), encoder_outputs)

            outputs[:, t] = output
            attentions[:, t] = attention

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs, attentions


def visualize_attention_in_translation():
    """
    Visualize attention weights in machine translation.
    """
    print("\n" + "="*70)
    print("ATTENTION VISUALIZATION IN TRANSLATION")
    print("="*70)

    # Simulate attention weights for translation
    source = ['The', 'cat', 'is', 'on', 'the', 'mat']
    target = ['Le', 'chat', 'est', 'sur', 'le', 'tapis']

    # Simulated attention weights (target x source)
    # Each row = target word, each column = source word
    attention = np.array([
        [0.6, 0.3, 0.05, 0.02, 0.02, 0.01],  # "Le" attends to "The"
        [0.1, 0.7, 0.1, 0.05, 0.03, 0.02],   # "chat" attends to "cat"
        [0.05, 0.1, 0.7, 0.1, 0.03, 0.02],   # "est" attends to "is"
        [0.02, 0.05, 0.1, 0.6, 0.2, 0.03],   # "sur" attends to "on"
        [0.1, 0.05, 0.05, 0.1, 0.6, 0.1],    # "le" attends to "the"
        [0.05, 0.05, 0.05, 0.05, 0.1, 0.7],  # "tapis" attends to "mat"
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap
    im = axes[0].imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(source)))
    axes[0].set_yticks(range(len(target)))
    axes[0].set_xticklabels(source, fontsize=11)
    axes[0].set_yticklabels(target, fontsize=11)
    axes[0].set_xlabel('Source (English)', fontsize=12, weight='bold')
    axes[0].set_ylabel('Target (French)', fontsize=12, weight='bold')
    axes[0].set_title('Attention Weights Heatmap', fontsize=13, weight='bold')

    # Add values
    for i in range(len(target)):
        for j in range(len(source)):
            text = axes[0].text(j, i, f'{attention[i, j]:.2f}',
                               ha="center", va="center",
                               color="white" if attention[i, j] > 0.5 else "black",
                               fontsize=9, weight='bold')

    plt.colorbar(im, ax=axes[0], label='Attention Weight')

    # Line plot for specific target word
    target_word_idx = 3  # "sur"
    axes[1].plot(range(len(source)), attention[target_word_idx], 'o-',
                linewidth=3, markersize=10, color='steelblue', label=f'"{target[target_word_idx]}"')
    axes[1].set_xticks(range(len(source)))
    axes[1].set_xticklabels(source, fontsize=11)
    axes[1].set_xlabel('Source Word', fontsize=12)
    axes[1].set_ylabel('Attention Weight', fontsize=12)
    axes[1].set_title(f'Attention for "{target[target_word_idx]}"', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    axes[1].legend()

    # Highlight max
    max_idx = attention[target_word_idx].argmax()
    axes[1].bar(max_idx, attention[target_word_idx][max_idx],
               alpha=0.3, color='red', edgecolor='darkred', linewidth=2)
    axes[1].text(max_idx, attention[target_word_idx][max_idx] + 0.05,
                f'Focuses on\n"{source[max_idx]}"',
                ha='center', fontsize=10, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('week4_attention_translation.png', dpi=150)
    plt.show()

    print("\n🔍 OBSERVATIONS:")
    print("-" * 70)
    print("• Diagonal pattern: Words align (The→Le, cat→chat, mat→tapis)")
    print("• Model learns word alignment automatically!")
    print("• Attention is interpretable: See what model focuses on")
    print("• Articles (the, le) have distributed attention")
    print("-" * 70)

    print("\n✓ Attention makes models interpretable and effective!")

visualize_attention_in_translation()
```

### 4.6 Self-Attention

**Key Innovation:** Apply attention within a single sequence!

**Use case:** Let each word attend to all other words in the same sentence

**Example:**

```
"The animal didn't cross the street because it was too tired"
```

What does "it" refer to? → "animal" (not "street")

Self-attention helps model capture this!

```python
class SelfAttention(nn.Module):
    """
    Self-Attention layer.

    Each position attends to all positions in the same sequence.
    """

    def __init__(self, embed_dim, num_heads=1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional mask for padding

        Returns:
            output: [batch, seq_len, embed_dim]
            attention: [batch, num_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, embed_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Now: [batch, num_heads, seq_len, head_dim]

        # Calculate attention scores
        # Q @ K^T / sqrt(d_k)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        # [batch, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Attention weights
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, V)  # [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_dim]
        out = out.view(batch_size, seq_len, self.embed_dim)  # [batch, seq_len, embed_dim]

        # Final linear projection
        out = self.W_o(out)

        return out, attention


def demonstrate_self_attention():
    """
    Demonstrate self-attention mechanism.
    """
    print("\n" + "="*70)
    print("SELF-ATTENTION DEMONSTRATION")
    print("="*70)

    # Parameters
    batch_size = 1
    seq_len = 6
    embed_dim = 8

    # Create self-attention layer
    self_attn = SelfAttention(embed_dim, num_heads=1)

    # Random input sequence
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"\n📊 Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Num heads: 1")

    # Forward pass
    output, attention = self_attn(x)

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Attention matrix
    attn_matrix = attention[0, 0].detach().numpy()  # [seq_len, seq_len]

    im = axes[0].imshow(attn_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_xlabel('Key Position', fontsize=12)
    axes[0].set_ylabel('Query Position', fontsize=12)
    axes[0].set_title('Self-Attention Matrix', fontsize=13, weight='bold')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))

    # Add values
    for i in range(seq_len):
        for j in range(seq_len):
            text = axes[0].text(j, i, f'{attn_matrix[i, j]:.2f}',
                               ha="center", va="center",
                               color="white" if attn_matrix[i, j] > 0.5 else "black",
                               fontsize=9, weight='bold')

    plt.colorbar(im, ax=axes[0], label='Attention Weight')

    # Attention for specific position
    pos = 2
    axes[1].bar(range(seq_len), attn_matrix[pos], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Position', fontsize=12)
    axes[1].set_ylabel('Attention Weight', fontsize=12)
    axes[1].set_title(f'Self-Attention from Position {pos}', fontsize=13, weight='bold')
    axes[1].set_xticks(range(seq_len))
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Highlight self-attention (diagonal)
    axes[1].bar(pos, attn_matrix[pos, pos], alpha=0.9, color='red', edgecolor='darkred', linewidth=3)
    axes[1].text(pos, attn_matrix[pos, pos] + 0.05, 'Self',
                ha='center', fontsize=10, weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('week4_self_attention.png', dpi=150)
    plt.show()

    print("\n🔑 SELF-ATTENTION KEY POINTS:")
    print("-" * 70)
    print("1. Query, Key, Value:")
    print("   • Q = W_q @ x  (What I'm looking for)")
    print("   • K = W_k @ x  (What I have to offer)")
    print("   • V = W_v @ x  (What information I carry)")
    print()
    print("2. Attention Score:")
    print("   • score(Q, K) = Q @ K^T / sqrt(d_k)")
    print("   • Scaled dot product")
    print("   • Scaling prevents gradient issues")
    print()
    print("3. Self-Attention Matrix:")
    print("   • Each position attends to all positions")
    print("   • Including itself (diagonal)")
    print("   • Captures relationships within sequence")
    print()
    print("4. Differences from Seq2Seq Attention:")
    print("   • Seq2Seq: Decoder attends to encoder")
    print("   • Self: Sequence attends to itself")
    print("   • Both use similar math!")
    print("-" * 70)

demonstrate_self_attention()
```

### 4.7 Key Takeaways from Day 4

✅ **Attention Mechanism**

- Solves context vector bottleneck
- Decoder attends to ALL encoder states
- Attention weights: Where to focus
- Dynamic context at each step

✅ **Bahdanau Attention (Additive)**
$$e_{ti} = v^T \tanh(W_1 s_t + W_2 h_i)$$

- Additive scoring function
- Learnable parameters: W₁, W₂, v

✅ **Luong Attention (Multiplicative)**

- Dot: $s_t^T h_i$ (fastest, no params)
- General: $s_t^T W h_i$ (good balance)
- Concat: Similar to Bahdanau

✅ **Self-Attention**

- Attention within single sequence
- Each token attends to all tokens
- Query, Key, Value projections
- Foundation for Transformers!

✅ **Benefits**

- No fixed-size bottleneck
- Better long-sequence performance
- Interpretable (visualize attention)
- State-of-the-art results

**Tomorrow:** Transformers - "Attention is All You Need"!

---

_End of Day 4. Total time: 6-8 hours._

---

<a name="day-5"></a>

## 📅 Day 5: Transformers and Modern Architectures

> "Attention is all you need." - Vaswani et al., 2017

### 5.1 From RNNs to Transformers

**Problem with RNNs (even with attention):**

- Sequential processing (can't parallelize)
- Long-range dependencies still challenging
- Slow training on long sequences

**Solution: TRANSFORMERS**

- Remove recurrence entirely
- Use only attention (self-attention)
- Fully parallelizable
- Better at capturing long-range dependencies

```python
def visualize_rnn_vs_transformer():
    """
    Compare RNN and Transformer architectures.
    """
    print("="*70)
    print("RNN vs TRANSFORMER ARCHITECTURE")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # RNN
    ax = axes[0]
    ax.set_title('RNN: Sequential Processing', fontsize=14, weight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    words = ['The', 'cat', 'sat', 'on', 'mat']
    x_pos = np.linspace(2, 8, len(words))

    for i, (x, word) in enumerate(zip(x_pos, words)):
        # Input
        ax.add_patch(plt.Circle((x, 8.5), 0.3, color='lightblue', ec='blue', linewidth=2))
        ax.text(x, 7.8, word, ha='center', fontsize=10, weight='bold')

        # RNN cell
        ax.add_patch(plt.Rectangle((x-0.3, 6), 0.6, 0.8, facecolor='lightgreen', edgecolor='green', linewidth=2))
        ax.text(x, 6.4, 'RNN', ha='center', fontsize=9, weight='bold')

        # Sequential connection
        if i < len(words) - 1:
            ax.annotate('', xy=(x_pos[i+1]-0.3, 6.4), xytext=(x+0.3, 6.4),
                       arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
            ax.text((x+x_pos[i+1])/2, 6.8, 'Sequential', ha='center', fontsize=8,
                   color='purple', style='italic')

        # Output
        ax.add_patch(plt.Circle((x, 4.5), 0.25, color='lightyellow', ec='orange', linewidth=2))
        ax.arrow(x, 6, 0, -1.2, head_width=0.15, head_length=0.15, fc='orange', ec='orange', linewidth=2)

    ax.text(5, 2.5, '❌ Must process sequentially', ha='center', fontsize=11, color='red', weight='bold')
    ax.text(5, 2, '❌ Slow on long sequences', ha='center', fontsize=10, color='red')
    ax.text(5, 1.5, '❌ Hard to parallelize', ha='center', fontsize=10, color='red')

    # Transformer
    ax = axes[1]
    ax.set_title('Transformer: Parallel Processing', fontsize=14, weight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    for i, (x, word) in enumerate(zip(x_pos, words)):
        # Input
        ax.add_patch(plt.Circle((x, 8.5), 0.3, color='lightblue', ec='blue', linewidth=2))
        ax.text(x, 7.8, word, ha='center', fontsize=10, weight='bold')

        # Self-attention block
        ax.add_patch(plt.Rectangle((x-0.3, 6), 0.6, 0.8, facecolor='lightcoral', edgecolor='red', linewidth=2))
        ax.text(x, 6.4, 'Attn', ha='center', fontsize=9, weight='bold')

        # All-to-all connections
        for j, x2 in enumerate(x_pos):
            if i != j:
                alpha = 0.15
                ax.plot([x, x2], [6.4, 6.4], 'g-', alpha=alpha, linewidth=1)

        # Output
        ax.add_patch(plt.Circle((x, 4.5), 0.25, color='lightyellow', ec='orange', linewidth=2))
        ax.arrow(x, 6, 0, -1.2, head_width=0.15, head_length=0.15, fc='orange', ec='orange', linewidth=2)

    ax.text(5, 5.5, 'Self-Attention: All positions communicate', ha='center',
           fontsize=9, color='green', style='italic')

    ax.text(5, 2.5, '✅ Fully parallelizable', ha='center', fontsize=11, color='green', weight='bold')
    ax.text(5, 2, '✅ Fast training', ha='center', fontsize=10, color='green')
    ax.text(5, 1.5, '✅ Better long-range dependencies', ha='center', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('week4_rnn_vs_transformer.png', dpi=150)
    plt.show()

    print("\n📊 RNN vs TRANSFORMER:")
    print("-" * 70)
    print(f"{'Property':<25} {'RNN':<25} {'Transformer':<25}")
    print("-" * 70)
    print(f"{'Processing':<25} {'Sequential':<25} {'Parallel':<25}")
    print(f"{'Speed (Training)':<25} {'Slow':<25} {'Fast':<25}")
    print(f"{'Long Dependencies':<25} {'Challenging':<25} {'Excellent':<25}")
    print(f"{'Memory':<25} {'O(1) per step':<25} {'O(n²) attention':<25}")
    print(f"{'Interpretability':<25} {'Limited':<25} {'Attention maps':<25}")
    print(f"{'Parallelization':<25} {'No (sequential)':<25} {'Yes (all tokens)':<25}")
    print("-" * 70)

visualize_rnn_vs_transformer()
```

### 5.2 Scaled Dot-Product Attention

**Core building block of Transformers**

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:

- $Q$ = Queries matrix
- $K$ = Keys matrix
- $V$ = Values matrix
- $d_k$ = Dimension of keys (for scaling)

**Why scaling?** Prevent dot products from growing too large (which makes softmax gradients small)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.

    Args:
        Q: Queries [batch, num_heads, seq_len, d_k]
        K: Keys [batch, num_heads, seq_len, d_k]
        V: Values [batch, num_heads, seq_len, d_v]
        mask: Optional mask [batch, 1, 1, seq_len]

    Returns:
        output: [batch, num_heads, seq_len, d_v]
        attention_weights: [batch, num_heads, seq_len, seq_len]
    """
    d_k = Q.shape[-1]

    # Calculate attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)  # [batch, heads, seq, seq]

    # Apply mask (for padding or future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def demonstrate_scaled_attention():
    """
    Demonstrate scaled dot-product attention.
    """
    print("\n" + "="*70)
    print("SCALED DOT-PRODUCT ATTENTION")
    print("="*70)

    # Parameters
    batch_size = 2
    num_heads = 1
    seq_len = 5
    d_k = 8
    d_v = 8

    # Create Q, K, V
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_v)

    print(f"\n📊 Input:")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  d_k (key dim): {d_k}")

    # Apply attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")

    # Show effect of scaling
    print(f"\n🔍 WHY SCALING MATTERS:")

    # Without scaling
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))

    # With scaling
    scores_scaled = scores_unscaled / np.sqrt(d_k)

    print(f"  Unscaled scores - mean: {scores_unscaled.mean():.4f}, std: {scores_unscaled.std():.4f}")
    print(f("  Scaled scores - mean: {scores_scaled.mean():.4f}, std: {scores_scaled.std():.4f}")
    print(f"  → Scaling keeps scores in reasonable range!")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sample = 0
    head = 0

    # Q·K^T (unscaled)
    im1 = axes[0].imshow(scores_unscaled[sample, head].detach().numpy(), cmap='coolwarm', aspect='auto')
    axes[0].set_xlabel('Key Position', fontsize=11)
    axes[0].set_ylabel('Query Position', fontsize=11)
    axes[0].set_title('Q·K^T (Unscaled)', fontsize=12, weight='bold')
    plt.colorbar(im1, ax=axes[0])

    # Scaled
    im2 = axes[1].imshow(scores_scaled[sample, head].detach().numpy(), cmap='coolwarm', aspect='auto')
    axes[1].set_xlabel('Key Position', fontsize=11)
    axes[1].set_ylabel('Query Position', fontsize=11)
    axes[1].set_title(f'Q·K^T / √{d_k} (Scaled)', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=axes[1])

    # Attention weights (after softmax)
    im3 = axes[2].imshow(attention_weights[sample, head].detach().numpy(), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[2].set_xlabel('Key Position', fontsize=11)
    axes[2].set_ylabel('Query Position', fontsize=11)
    axes[2].set_title('Attention Weights (Softmax)', fontsize=12, weight='bold')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('week4_scaled_attention.png', dpi=150)
    plt.show()

    print("\n✓ Scaled dot-product attention complete!")
    print("  → Scaling prevents saturation of softmax")
    print("  → Maintains stable gradients")

demonstrate_scaled_attention()
```

### 5.3 Multi-Head Attention

**Key Idea:** Run attention multiple times in parallel with different learned projections

**Why?** Allows model to jointly attend to different aspects:

- Head 1: Syntax
- Head 2: Semantics
- Head 3: Long-range dependencies
- etc.

**Formula:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.

    The cornerstone of Transformer architecture.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: Optional mask

        Returns:
            output: [batch, seq_len, d_model]
            attention: [batch, num_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]

        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        # [batch, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        x = torch.matmul(attention, V)  # [batch, num_heads, seq_len, d_k]

        # Concatenate heads
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, -1, self.d_model)  # [batch, seq_len, d_model]

        # Final linear projection
        output = self.W_o(x)

        return output, attention


def demonstrate_multi_head_attention():
    """
    Demonstrate multi-head attention with visualization.
    """
    print("\n" + "="*70)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("="*70)

    # Parameters
    batch_size = 1
    seq_len = 8
    d_model = 16
    num_heads = 4

    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n📊 Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (per head): {d_model // num_heads}")
    print(f"  Sequence length: {seq_len}")

    # Forward pass (self-attention: Q=K=V=x)
    output, attention = mha(x, x, x)

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")

    # Visualize different heads
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    attention_np = attention[0].detach().numpy()  # [num_heads, seq_len, seq_len]

    for head in range(num_heads):
        im = axes[head].imshow(attention_np[head], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[head].set_xlabel('Key Position', fontsize=11)
        axes[head].set_ylabel('Query Position', fontsize=11)
        axes[head].set_title(f'Head {head+1} Attention', fontsize=12, weight='bold')
        axes[head].set_xticks(range(seq_len))
        axes[head].set_yticks(range(seq_len))
        plt.colorbar(im, ax=axes[head])

    plt.tight_layout()
    plt.savefig('week4_multi_head_attention.png', dpi=150)
    plt.show()

    print("\n🔑 MULTI-HEAD ATTENTION INSIGHTS:")
    print("-" * 70)
    print("• Different heads learn different patterns:")
    print("  - Some focus on local context (nearby words)")
    print("  - Some focus on long-range dependencies")
    print("  - Some capture syntax, others semantics")
    print()
    print("• Attention patterns differ across heads")
    print("• Model can attend to multiple aspects simultaneously")
    print("• More expressive than single attention")
    print("-" * 70)

    print("\n✓ Multi-head attention provides richer representations!")

demonstrate_multi_head_attention()
```

### 5.4 Positional Encoding

**Problem:** Self-attention has NO notion of position!

- "The cat sat" and "sat cat The" produce same representation

**Solution:** Add positional information to embeddings

**Sinusoidal Positional Encoding:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.

    Adds position information to token embeddings.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_positional_encoding():
    """
    Visualize positional encoding patterns.
    """
    print("\n" + "="*70)
    print("POSITIONAL ENCODING VISUALIZATION")
    print("="*70)

    d_model = 128
    max_len = 100

    pe = PositionalEncoding(d_model, max_len)

    # Get positional encodings
    pos_enc = pe.pe[0, :max_len, :].numpy()  # [max_len, d_model]

    print(f"\n📊 Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    print(f"  Encoding shape: {pos_enc.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Heatmap of all encodings
    im1 = axes[0, 0].imshow(pos_enc.T, cmap='coolwarm', aspect='auto')
    axes[0, 0].set_xlabel('Position', fontsize=11)
    axes[0, 0].set_ylabel('Dimension', fontsize=11)
    axes[0, 0].set_title('Positional Encoding Heatmap', fontsize=12, weight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # Encoding for specific positions
    positions = [0, 10, 25, 50]
    for pos in positions:
        axes[0, 1].plot(pos_enc[pos, :64], label=f'Position {pos}', linewidth=2, alpha=0.7)

    axes[0, 1].set_xlabel('Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Value', fontsize=11)
    axes[0, 1].set_title('Encoding Patterns (First 64 dims)', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Frequency analysis (different dimensions)
    dims = [0, 10, 50, 100]
    for dim in dims:
        axes[1, 0].plot(pos_enc[:50, dim], label=f'Dim {dim}', linewidth=2, alpha=0.7)

    axes[1, 0].set_xlabel('Position', fontsize=11)
    axes[1, 0].set_ylabel('Value', fontsize=11)
    axes[1, 0].set_title('Position Signals (Different Frequencies)', fontsize=12, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Similarity between positions
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(pos_enc[:50])

    im2 = axes[1, 1].imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Position', fontsize=11)
    axes[1, 1].set_ylabel('Position', fontsize=11)
    axes[1, 1].set_title('Position Similarity Matrix', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=axes[1, 1], label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig('week4_positional_encoding.png', dpi=150)
    plt.show()

    print("\n🔑 POSITIONAL ENCODING INSIGHTS:")
    print("-" * 70)
    print("1. Sine/Cosine Functions:")
    print("   • Different frequencies for different dimensions")
    print("   • Low dims: slow oscillation (capture global position)")
    print("   • High dims: fast oscillation (capture local position)")
    print()
    print("2. Properties:")
    print("   • Deterministic (not learned)")
    print("   • Unique encoding for each position")
    print("   • Smooth transitions between positions")
    print("   • Can extrapolate to longer sequences")
    print()
    print("3. Alternative:")
    print("   • Learned positional embeddings (like BERT)")
    print("   • Relative position encodings")
    print("-" * 70)

    print("\n✓ Positional encoding gives Transformer sense of order!")

visualize_positional_encoding()
```

### 5.5 Transformer Block

**Complete Transformer block:**

1. Multi-Head Self-Attention
2. Add & Norm (Residual connection + Layer Normalization)
3. Feed-Forward Network
4. Add & Norm

```python
class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.

    Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Multi-head self-attention with residual
        attn_output, attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention


def demonstrate_transformer_block():
    """
    Demonstrate a complete Transformer block.
    """
    print("\n" + "="*70)
    print("TRANSFORMER BLOCK DEMONSTRATION")
    print("="*70)

    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # Create Transformer block
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.1)

    print(f"\n📊 Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  Sequence length: {seq_len}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {total_params:,}")

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n📥 Input shape: {x.shape}")

    # Forward pass
    output, attention = block(x)

    print(f"\n📤 Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  ✓ Shape preserved through block!")

    # Visualize architecture
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Transformer Block Architecture', fontsize=16, weight='bold')

    # Input
    ax.add_patch(plt.Rectangle((3, 9), 4, 0.5, facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(5, 9.25, 'Input: [batch, seq_len, d_model]', ha='center', fontsize=10, weight='bold')

    # Multi-Head Attention
    ax.arrow(5, 9, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.add_patch(plt.Rectangle((2.5, 7.5), 5, 1, facecolor='#FFE5B4', edgecolor='orange', linewidth=3))
    ax.text(5, 8.2, 'Multi-Head', ha='center', fontsize=11, weight='bold')
    ax.text(5, 7.8, 'Self-Attention', ha='center', fontsize=11, weight='bold')

    # Add & Norm 1
    ax.arrow(5, 7.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.add_patch(plt.Rectangle((3, 6.5), 4, 0.6, facecolor='#B4E5FF', edgecolor='blue', linewidth=2))
    ax.text(5, 6.8, 'Add & Norm', ha='center', fontsize=10, weight='bold')

    # Residual connection
    ax.arrow(1.5, 9.25, 0, -2.5, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2, linestyle='--')
    ax.arrow(1.5, 6.75, 1.3, 0, head_width=0.1, head_length=0.15, fc='red', ec='red', linewidth=2, linestyle='--')
    ax.text(1, 7.5, 'Residual', ha='center', fontsize=9, color='red', rotation=90)

    # Feed-Forward
    ax.arrow(5, 6.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.add_patch(plt.Rectangle((2.5, 4.5), 5, 1.5, facecolor='#D4F4DD', edgecolor='green', linewidth=3))
    ax.text(5, 5.6, 'Feed-Forward', ha='center', fontsize=11, weight='bold')
    ax.text(5, 5.2, 'Network', ha='center', fontsize=11, weight='bold')
    ax.text(5, 4.8, f'(d_model → {d_ff} → d_model)', ha='center', fontsize=9, style='italic')

    # Add & Norm 2
    ax.arrow(5, 4.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.add_patch(plt.Rectangle((3, 3.5), 4, 0.6, facecolor='#B4E5FF', edgecolor='blue', linewidth=2))
    ax.text(5, 3.8, 'Add & Norm', ha='center', fontsize=10, weight='bold')

    # Residual connection 2
    ax.arrow(8.5, 6.75, 0, -3.1, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2, linestyle='--')
    ax.arrow(8.5, 3.65, -1.3, 0, head_width=0.1, head_length=0.15, fc='red', ec='red', linewidth=2, linestyle='--')
    ax.text(9, 5, 'Residual', ha='center', fontsize=9, color='red', rotation=90)

    # Output
    ax.arrow(5, 3.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.add_patch(plt.Rectangle((3, 2.5), 4, 0.5, facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(5, 2.75, 'Output: [batch, seq_len, d_model]', ha='center', fontsize=10, weight='bold')

    # Annotations
    ax.text(5, 1.5, 'Key Components:', ha='center', fontsize=12, weight='bold')
    ax.text(5, 1, '• Multi-Head Attention: Capture relationships', ha='center', fontsize=9)
    ax.text(5, 0.7, '• Feed-Forward: Process each position independently', ha='center', fontsize=9)
    ax.text(5, 0.4, '• Residual Connections: Enable deep networks', ha='center', fontsize=9)
    ax.text(5, 0.1, '• Layer Norm: Stabilize training', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('week4_transformer_block.png', dpi=150)
    plt.show()

    print("\n✓ Transformer block: Self-attention + Feed-forward + Residuals!")

demonstrate_transformer_block()
```

### 5.6 Complete Transformer Architecture

**Full Transformer (Original paper):**

- **Encoder:** Stack of N encoder blocks
- **Decoder:** Stack of N decoder blocks
- **Output:** Linear + Softmax

```python
def visualize_complete_transformer():
    """
    Visualize the complete Transformer architecture.
    """
    print("\n" + "="*70)
    print("COMPLETE TRANSFORMER ARCHITECTURE")
    print("="*70)

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Transformer Architecture (Vaswani et al., 2017)', fontsize=16, weight='bold')

    # Encoder side (left)
    enc_x = 2.5

    # Input
    ax.add_patch(plt.Rectangle((enc_x-0.8, 9), 1.6, 0.4, facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(enc_x, 9.2, 'Input\nEmbedding', ha='center', fontsize=9, weight='bold')

    # Positional encoding
    ax.add_patch(plt.Rectangle((enc_x-0.8, 8.3), 1.6, 0.4, facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(enc_x, 8.5, 'Positional\nEncoding', ha='center', fontsize=9, weight='bold')
    ax.arrow(enc_x, 8.9, 0, -0.15, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)

    # Encoder blocks (stack of N)
    for i in range(3):
        y = 7 - i * 1.5

        if i < 2:
            ax.add_patch(plt.Rectangle((enc_x-0.9, y), 1.8, 1.2, facecolor='#E8F5E9', edgecolor='green', linewidth=2))
            ax.text(enc_x, y + 1, 'Multi-Head', ha='center', fontsize=8, weight='bold')
            ax.text(enc_x, y + 0.8, 'Attention', ha='center', fontsize=8, weight='bold')
            ax.text(enc_x, y + 0.5, 'Add & Norm', ha='center', fontsize=7)
            ax.text(enc_x, y + 0.2, 'Feed Forward', ha='center', fontsize=7, weight='bold')
            ax.text(enc_x, y, 'Add & Norm', ha='center', fontsize=7)

            if i < 1:
                ax.arrow(enc_x, y, 0, -0.25, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
        else:
            # Show stacking
            ax.text(enc_x, y + 0.5, '...', ha='center', fontsize=20, weight='bold')
            ax.text(enc_x, y + 0.1, f'×N layers', ha='center', fontsize=9, style='italic')

    ax.text(enc_x, 2.5, 'ENCODER', ha='center', fontsize=13, weight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Decoder side (right)
    dec_x = 7.5

    # Output (shifted right)
    ax.add_patch(plt.Rectangle((dec_x-0.8, 9), 1.6, 0.4, facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax.text(dec_x, 9.2, 'Output\nEmbedding', ha='center', fontsize=9, weight='bold')

    # Positional encoding
    ax.add_patch(plt.Rectangle((dec_x-0.8, 8.3), 1.6, 0.4, facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(dec_x, 8.5, 'Positional\nEncoding', ha='center', fontsize=9, weight='bold')
    ax.arrow(dec_x, 8.9, 0, -0.15, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)

    # Decoder blocks (stack of N)
    for i in range(3):
        y = 7 - i * 1.8

        if i < 2:
            ax.add_patch(plt.Rectangle((dec_x-0.9, y), 1.8, 1.5, facecolor='#FFEBEE', edgecolor='red', linewidth=2))
            ax.text(dec_x, y + 1.35, 'Masked', ha='center', fontsize=7)
            ax.text(dec_x, y + 1.15, 'Multi-Head Attention', ha='center', fontsize=7, weight='bold')
            ax.text(dec_x, y + 0.95, 'Add & Norm', ha='center', fontsize=6)
            ax.text(dec_x, y + 0.7, 'Cross-Attention', ha='center', fontsize=7, weight='bold')
            ax.text(dec_x, y + 0.5, '(Encoder-Decoder)', ha='center', fontsize=6)
            ax.text(dec_x, y + 0.3, 'Add & Norm', ha='center', fontsize=6)
            ax.text(dec_x, y + 0.05, 'Feed Forward', ha='center', fontsize=7, weight='bold')

            # Cross-attention connection from encoder
            ax.arrow(enc_x + 1, 5.5 - i * 1.8, dec_x - enc_x - 2, 0,
                    head_width=0.1, head_length=0.15, fc='purple', ec='purple', linewidth=2, linestyle='--')

            if i < 1:
                ax.arrow(dec_x, y, 0, -0.25, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
        else:
            ax.text(dec_x, y + 0.7, '...', ha='center', fontsize=20, weight='bold')
            ax.text(dec_x, y + 0.3, f'×N layers', ha='center', fontsize=9, style='italic')

    # Output layers
    ax.add_patch(plt.Rectangle((dec_x-0.8, 1.5), 1.6, 0.4, facecolor='#FFF9C4', edgecolor='orange', linewidth=2))
    ax.text(dec_x, 1.7, 'Linear', ha='center', fontsize=9, weight='bold')
    ax.arrow(dec_x, 2.5, 0, -0.55, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)

    ax.add_patch(plt.Rectangle((dec_x-0.8, 0.8), 1.6, 0.4, facecolor='#FFE0B2', edgecolor='orange', linewidth=2))
    ax.text(dec_x, 1, 'Softmax', ha='center', fontsize=9, weight='bold')
    ax.arrow(dec_x, 1.5, 0, -0.25, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)

    ax.add_patch(plt.Rectangle((dec_x-0.8, 0.2), 1.6, 0.3, facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(dec_x, 0.35, 'Output Probabilities', ha='center', fontsize=8, weight='bold')
    ax.arrow(dec_x, 0.8, 0, -0.25, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)

    ax.text(dec_x, 2.3, 'DECODER', ha='center', fontsize=13, weight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig('week4_complete_transformer.png', dpi=150)
    plt.show()

    print("\n🏗️  TRANSFORMER ARCHITECTURE:")
    print("-" * 70)
    print("ENCODER:")
    print("  1. Input Embedding + Positional Encoding")
    print("  2. N × Encoder Blocks:")
    print("     • Multi-Head Self-Attention")
    print("     • Add & Norm")
    print("     • Feed-Forward Network")
    print("     • Add & Norm")
    print()
    print("DECODER:")
    print("  1. Output Embedding + Positional Encoding")
    print("  2. N × Decoder Blocks:")
    print("     • Masked Multi-Head Self-Attention (can't see future)")
    print("     • Add & Norm")
    print("     • Cross-Attention (attend to encoder)")
    print("     • Add & Norm")
    print("     • Feed-Forward Network")
    print("     • Add & Norm")
    print("  3. Linear + Softmax → Probabilities")
    print()
    print("KEY INNOVATIONS:")
    print("  • No recurrence (fully parallel)")
    print("  • Self-attention for relationships")
    print("  • Positional encoding for order")
    print("  • Multi-head for multiple perspectives")
    print("  • Residual connections for deep networks")
    print("-" * 70)

    print("\n📊 STANDARD CONFIGURATION (Transformer-base):")
    print("  • N = 6 (6 encoder + 6 decoder layers)")
    print("  • d_model = 512 (embedding dimension)")
    print("  • num_heads = 8 (8 attention heads)")
    print("  • d_ff = 2048 (feed-forward dimension)")
    print("  • dropout = 0.1")
    print("  • Total parameters: ~65M")

    print("\n📊 LARGER MODELS:")
    print("  • Transformer-big: d_model=1024, N=6, ~213M params")
    print("  • GPT-3: d_model=12288, N=96, ~175B params")
    print("  • GPT-4: Estimated ~1.7T params (mixture of experts)")

    print("\n✓ Transformer: The foundation of modern NLP!")

visualize_complete_transformer()
```

### 5.7 BERT vs GPT

**Two major Transformer variants:**

**BERT** (Bidirectional Encoder Representations from Transformers)

- Uses encoder only
- Bidirectional context
- Masked language modeling
- Good for: Classification, QA, NER

**GPT** (Generative Pre-trained Transformer)

- Uses decoder only
- Unidirectional (left-to-right)
- Causal language modeling
- Good for: Text generation, completion

```python
def compare_bert_gpt():
    """
    Compare BERT and GPT architectures.
    """
    print("\n" + "="*70)
    print("BERT vs GPT COMPARISON")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # BERT (Encoder-only)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('BERT: Encoder-Only (Bidirectional)', fontsize=14, weight='bold')

    # Input
    input_text = ['[CLS]', 'The', '[MASK]', 'sat', '[SEP]']
    x_pos = np.linspace(2, 8, len(input_text))

    for i, (x, token) in enumerate(zip(x_pos, input_text)):
        ax.add_patch(plt.Circle((x, 8.5), 0.35, color='lightblue', ec='blue', linewidth=2))
        ax.text(x, 7.7, token, ha='center', fontsize=10, weight='bold')

    # Bidirectional attention
    ax.add_patch(plt.Rectangle((1.5, 5.5), 7, 1.5, facecolor='lightgreen', edgecolor='green', linewidth=3))
    ax.text(5, 6.6, 'Bidirectional', ha='center', fontsize=11, weight='bold')
    ax.text(5, 6.2, 'Self-Attention', ha='center', fontsize=11, weight='bold')
    ax.text(5, 5.8, '(All tokens see all tokens)', ha='center', fontsize=9, style='italic')

    # Show all-to-all connections
    for i, x1 in enumerate(x_pos):
        for j, x2 in enumerate(x_pos):
            if i != j:
                ax.plot([x1, x2], [8, 6.8], 'g-', alpha=0.1, linewidth=1)

    # Output
    for i, (x, token) in enumerate(zip(x_pos, input_text)):
        ax.add_patch(plt.Circle((x, 3), 0.35, color='lightyellow', ec='orange', linewidth=2))
        if token == '[MASK]':
            ax.text(x, 2.2, '"cat"', ha='center', fontsize=10, weight='bold', color='red')
        else:
            ax.text(x, 2.2, token, ha='center', fontsize=9)

    ax.text(5, 1, '✅ Predicts masked word: "cat"', ha='center', fontsize=11, color='green', weight='bold')
    ax.text(5, 0.5, 'Training: Mask random tokens', ha='center', fontsize=10, style='italic')

    # GPT (Decoder-only)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('GPT: Decoder-Only (Unidirectional)', fontsize=14, weight='bold')

    # Input
    input_gpt = ['The', 'cat', 'sat', 'on', '___']

    for i, (x, token) in enumerate(zip(x_pos, input_gpt)):
        ax.add_patch(plt.Circle((x, 8.5), 0.35, color='lightcoral', ec='red', linewidth=2))
        ax.text(x, 7.7, token, ha='center', fontsize=10, weight='bold')

    # Causal (masked) attention
    ax.add_patch(plt.Rectangle((1.5, 5.5), 7, 1.5, facecolor='#FFE5B4', edgecolor='orange', linewidth=3))
    ax.text(5, 6.6, 'Causal', ha='center', fontsize=11, weight='bold')
    ax.text(5, 6.2, 'Self-Attention', ha='center', fontsize=11, weight='bold')
    ax.text(5, 5.8, '(Can only see past tokens)', ha='center', fontsize=9, style='italic')

    # Show causal connections (only to past)
    for i, x1 in enumerate(x_pos):
        for j, x2 in enumerate(x_pos[:i+1]):  # Only past and self
            ax.plot([x1, x2], [8, 6.8], 'orange', alpha=0.2, linewidth=1.5)

    # Output
    for i, (x, token) in enumerate(zip(x_pos, input_gpt)):
        ax.add_patch(plt.Circle((x, 3), 0.35, color='lightyellow', ec='orange', linewidth=2))
        if token == '___':
            ax.text(x, 2.2, '"the"', ha='center', fontsize=10, weight='bold', color='red')
        else:
            ax.text(x, 2.2, token, ha='center', fontsize=9)

    ax.text(5, 1, '✅ Predicts next word: "the"', ha='center', fontsize=11, color='green', weight='bold')
    ax.text(5, 0.5, 'Training: Predict next token', ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('week4_bert_vs_gpt.png', dpi=150)
    plt.show()

    print("\n📊 BERT vs GPT:")
    print("-" * 70)
    print(f"{'Property':<25} {'BERT':<30} {'GPT':<30}")
    print("-" * 70)
    print(f"{'Architecture':<25} {'Encoder only':<30} {'Decoder only':<30}")
    print(f"{'Attention':<25} {'Bidirectional':<30} {'Causal (Unidirectional)':<30}")
    print(f"{'Training Task':<25} {'Masked LM':<30} {'Next token prediction':<30}")
    print(f"{'Use Case':<25} {'Understanding':<30} {'Generation':<30}")
    print(f"{'Examples':<25} {'Classification, QA':<30} {'Text completion, chatbots':<30}")
    print(f"{'Can see future':<25} {'Yes (in training)':<30} {'No (causal masking)':<30}")
    print("-" * 70)

    print("\n💡 KEY DIFFERENCES:")
    print("  BERT:")
    print("    • Sees entire sentence (bidirectional)")
    print("    • Learns by filling in blanks ([MASK])")
    print("    • Better for understanding tasks")
    print("    • Example: 'Paris is the capital of [MASK]' → 'France'")
    print()
    print("  GPT:")
    print("    • Only sees past words (left-to-right)")
    print("    • Learns by predicting next word")
    print("    • Better for generation tasks")
    print("    • Example: 'Once upon a time' → generates full story")

    print("\n🔮 MODERN MODELS:")
    print("  • BERT → RoBERTa, ALBERT, DeBERTa")
    print("  • GPT → GPT-2, GPT-3, GPT-4, ChatGPT")
    print("  • Hybrid: T5, BART (encoder-decoder)")

compare_bert_gpt()
```

### 5.8 Key Takeaways from Day 5

✅ **Transformers**

- Replace recurrence with self-attention
- Fully parallelizable
- State-of-the-art on most NLP tasks
- Foundation of modern AI (GPT, BERT, etc.)

✅ **Core Components**

- Multi-Head Attention: Multiple perspectives
- Positional Encoding: Add position information
- Feed-Forward Networks: Process each position
- Residual Connections: Enable deep networks

✅ **Architecture**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

✅ **Variants**

- BERT: Encoder-only, bidirectional, understanding
- GPT: Decoder-only, causal, generation
- T5/BART: Encoder-decoder, both tasks

✅ **Why Transformers Win**

- Parallel processing (fast training)
- Long-range dependencies (attention)
- Scalable (GPT-3: 175B parameters)
- Transfer learning (pre-train, fine-tune)

**Tomorrow (Weekend):** Build a complete text generation engine!

---

_End of Day 5. Total time: 6-8 hours._

---

<a name="weekend-project"></a>

## 🎯 Weekend Project: Text Generation Engine

> "Build a complete text generation system from scratch."

### Project Overview

**Goal:** Create a production-ready text generation engine with multiple architectures and sampling strategies.

**What you'll build:**

1. Character-level RNN generator
2. Word-level LSTM generator with attention
3. Multiple sampling strategies (greedy, temperature, top-k, nucleus)
4. Training pipeline with monitoring
5. Interactive generation interface
6. Model comparison and evaluation

**Time:** 8-12 hours

---

### Part 1: Data Preparation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import string

class TextDataset:
    """
    Dataset for text generation.
    Supports both character-level and word-level.
    """

    def __init__(self, text, level='char', seq_length=100):
        """
        Args:
            text: Raw text string
            level: 'char' or 'word'
            seq_length: Length of input sequences
        """
        self.level = level
        self.seq_length = seq_length

        # Preprocessing
        if level == 'char':
            self.vocab = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)

            # Encode text
            self.data = [self.char_to_idx[ch] for ch in text]

        elif level == 'word':
            # Tokenize
            words = self._tokenize(text)

            # Build vocabulary
            word_counts = Counter(words)
            self.vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + \
                         [word for word, count in word_counts.most_common(10000)]

            self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
            self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)

            # Encode text
            self.data = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                        for word in words]

        print(f"📊 Dataset Statistics:")
        print(f"  Level: {level}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Total sequences: {len(self.data) - seq_length}")

    def _tokenize(self, text):
        """Simple word tokenization."""
        # Lowercase
        text = text.lower()
        # Add space around punctuation
        text = re.sub(f'([{string.punctuation}])', r' \1 ', text)
        # Split
        words = text.split()
        return words

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        """
        Returns:
            input_seq: [seq_length]
            target_seq: [seq_length] (shifted by 1)
        """
        input_seq = torch.tensor(self.data[idx:idx + self.seq_length])
        target_seq = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1])
        return input_seq, target_seq


def prepare_data(filename='shakespeare.txt', level='char', seq_length=100, batch_size=64):
    """
    Load and prepare text data.
    """
    print("="*70)
    print("DATA PREPARATION")
    print("="*70)

    # Load text
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Use sample text if file not found
        print(f"\n⚠️  File '{filename}' not found. Using sample text.")
        text = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """ * 50  # Repeat to have more data

    print(f"\n📖 Text loaded:")
    print(f"  Length: {len(text):,} characters")
    print(f"  Preview: {text[:200]}...")

    # Create dataset
    dataset = TextDataset(text, level=level, seq_length=seq_length)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"\n✓ Data ready!")
    print(f"  Train sequences: {train_size:,}")
    print(f"  Val sequences: {val_size:,}")

    return dataset, train_loader, val_loader


# Prepare data
dataset, train_loader, val_loader = prepare_data(level='char', seq_length=100, batch_size=128)
```

---

### Part 2: Character-Level RNN Generator

```python
class CharRNNGenerator(nn.Module):
    """
    Character-level RNN for text generation.

    Architecture:
    Input → Embedding → LSTM → Dropout → Linear → Output
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len] - input sequences
            hidden: Optional hidden state

        Returns:
            output: [batch, seq_len, vocab_size]
            hidden: Hidden state
        """
        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch, seq_len, hidden_dim]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Output
        output = self.fc(lstm_out)  # [batch, seq_len, vocab_size]

        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)


def train_char_rnn(model, train_loader, val_loader, dataset, epochs=20, lr=0.002):
    """
    Train character-level RNN.
    """
    print("\n" + "="*70)
    print("TRAINING CHARACTER-LEVEL RNN")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []

    print(f"\n🚀 Starting training...")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {train_loader.batch_size}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs, _ = model(inputs)

            # Calculate loss
            loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Generate sample text
        if (epoch + 1) % 5 == 0:
            sample = generate_text_char(model, dataset, 'To be', max_len=100, temperature=0.8)
            print(f"\n  Sample generation: {sample}\n")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('week4_char_rnn_training.png', dpi=150)
    plt.show()

    print("\n✓ Training complete!")

    return model, train_losses, val_losses


# Create and train model
char_model = CharRNNGenerator(
    vocab_size=dataset.vocab_size,
    embed_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3
)

print(f"\n📊 Model Architecture:")
print(char_model)
print(f"\nParameters: {sum(p.numel() for p in char_model.parameters()):,}")

# Train
char_model, train_losses, val_losses = train_char_rnn(
    char_model, train_loader, val_loader, dataset, epochs=30, lr=0.002
)
```

---

### Part 3: Sampling Strategies

```python
def generate_text_char(model, dataset, start_text, max_len=200,
                      temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate text using character-level model.

    Args:
        model: Trained model
        dataset: Dataset with vocabulary
        start_text: Starting string
        max_len: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling (0.0 = disabled)

    Returns:
        Generated text string
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode start text
    current_text = start_text
    input_seq = torch.tensor([dataset.char_to_idx.get(ch, 0) for ch in start_text]).unsqueeze(0).to(device)

    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass
            output, hidden = model(input_seq, hidden)

            # Get last time step
            logits = output[0, -1, :] / temperature  # [vocab_size]

            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)

            # Top-p (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()

            next_char = dataset.idx_to_char[next_char_idx]
            current_text += next_char

            # Update input
            input_seq = torch.tensor([[next_char_idx]]).to(device)

    return current_text


def demonstrate_sampling_strategies():
    """
    Demonstrate different sampling strategies.
    """
    print("\n" + "="*70)
    print("SAMPLING STRATEGIES COMPARISON")
    print("="*70)

    start_text = "To be"
    max_len = 150

    strategies = [
        ("Greedy (temperature=0.1)", {"temperature": 0.1, "top_k": 0, "top_p": 0.0}),
        ("Temperature=1.0", {"temperature": 1.0, "top_k": 0, "top_p": 0.0}),
        ("Temperature=1.5", {"temperature": 1.5, "top_k": 0, "top_p": 0.0}),
        ("Top-k (k=10)", {"temperature": 1.0, "top_k": 10, "top_p": 0.0}),
        ("Nucleus (p=0.9)", {"temperature": 1.0, "top_k": 0, "top_p": 0.9}),
    ]

    print(f"\n🔤 Start text: '{start_text}'\n")

    for name, params in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {name}")
        print(f"{'='*70}")

        generated = generate_text_char(char_model, dataset, start_text, max_len=max_len, **params)

        print(f"\n{generated}\n")

    print("\n💡 SAMPLING STRATEGIES EXPLAINED:")
    print("-" * 70)
    print("1. Greedy (Low Temperature):")
    print("   • Always picks most likely token")
    print("   • Deterministic, repetitive")
    print("   • Use for factual, consistent output")
    print()
    print("2. High Temperature:")
    print("   • More random sampling")
    print("   • Creative but can be incoherent")
    print("   • Use for diverse, creative output")
    print()
    print("3. Top-k Sampling:")
    print("   • Sample from top k most likely tokens")
    print("   • Balances diversity and quality")
    print("   • k=5-50 typically works well")
    print()
    print("4. Nucleus (Top-p) Sampling:")
    print("   • Sample from smallest set with cumulative prob ≥ p")
    print("   • Adaptive: varies number of candidates")
    print("   • p=0.9-0.95 recommended")
    print("   • Used by GPT-3")
    print("-" * 70)

demonstrate_sampling_strategies()
```

---

### Part 4: Word-Level LSTM with Attention

```python
class WordLSTMGenerator(nn.Module):
    """
    Word-level LSTM with attention for text generation.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)

        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len]
            hidden: Optional LSTM hidden state

        Returns:
            output: [batch, seq_len, vocab_size]
            hidden: LSTM hidden state
        """
        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch, seq_len, hidden_dim]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, seq_len, hidden_dim]

        # Residual connection
        combined = lstm_out + attn_out

        # Dropout
        combined = self.dropout(combined)

        # Output
        output = self.fc(combined)  # [batch, seq_len, vocab_size]

        return output, hidden


def generate_text_word(model, dataset, start_text, max_len=50, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate text using word-level model.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize start text
    words = dataset._tokenize(start_text)
    current_text = words.copy()

    # Encode
    input_seq = torch.tensor([dataset.word_to_idx.get(word, dataset.word_to_idx['<UNK>'])
                              for word in words]).unsqueeze(0).to(device)

    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            # Forward
            output, hidden = model(input_seq, hidden)

            # Get last time step
            logits = output[0, -1, :] / temperature

            # Top-k
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)

            # Top-p
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_word_idx = torch.multinomial(probs, 1).item()

            next_word = dataset.idx_to_word[next_word_idx]
            current_text.append(next_word)

            # Stop at sentence end
            if next_word in ['.', '!', '?']:
                break

            # Update input
            input_seq = torch.tensor([[next_word_idx]]).to(device)

    # Join words
    return ' '.join(current_text)


# Prepare word-level data
word_dataset, word_train_loader, word_val_loader = prepare_data(
    level='word', seq_length=50, batch_size=64
)

# Create word-level model
word_model = WordLSTMGenerator(
    vocab_size=word_dataset.vocab_size,
    embed_dim=256,
    hidden_dim=512,
    num_layers=2,
    dropout=0.3
)

print(f"\n📊 Word-Level Model:")
print(f"  Parameters: {sum(p.numel() for p in word_model.parameters()):,}")

# Train (similar to char model)
print("\n🚀 Training word-level model...")
print("  (Training code same as char model, adapted for word-level)")
```

---

### Part 5: Interactive Generation Interface

```python
class TextGeneratorInterface:
    """
    Interactive interface for text generation.
    """

    def __init__(self, char_model, word_model, char_dataset, word_dataset):
        self.char_model = char_model
        self.word_model = word_model
        self.char_dataset = char_dataset
        self.word_dataset = word_dataset

        self.char_model.eval()
        self.word_model.eval()

    def generate(self, start_text, model_type='char', max_len=200,
                temperature=1.0, top_k=0, top_p=0.0):
        """
        Generate text interactively.
        """
        print("\n" + "="*70)
        print(f"TEXT GENERATION - {model_type.upper()} LEVEL")
        print("="*70)

        print(f"\n📝 Start text: '{start_text}'")
        print(f"⚙️  Settings:")
        print(f"   • Max length: {max_len}")
        print(f"   • Temperature: {temperature}")
        print(f"   • Top-k: {top_k if top_k > 0 else 'disabled'}")
        print(f"   • Top-p: {top_p if top_p > 0 else 'disabled'}")

        print(f"\n🎯 Generating...\n")

        if model_type == 'char':
            result = generate_text_char(
                self.char_model, self.char_dataset, start_text,
                max_len=max_len, temperature=temperature, top_k=top_k, top_p=top_p
            )
        else:
            result = generate_text_word(
                self.word_model, self.word_dataset, start_text,
                max_len=max_len, temperature=temperature, top_k=top_k, top_p=top_p
            )

        print(f"📄 Generated text:")
        print("-" * 70)
        print(result)
        print("-" * 70)

        return result

    def compare_models(self, start_text, max_len=150):
        """
        Compare character-level and word-level generation.
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        # Character-level
        print(f"\n🔤 CHARACTER-LEVEL RNN:")
        print("-" * 70)
        char_result = generate_text_char(
            self.char_model, self.char_dataset, start_text, max_len=max_len, temperature=0.8
        )
        print(char_result)

        # Word-level
        print(f"\n\n📝 WORD-LEVEL LSTM + ATTENTION:")
        print("-" * 70)
        word_result = generate_text_word(
            self.word_model, self.word_dataset, start_text, max_len=max_len//3, temperature=0.8
        )
        print(word_result)

        print("\n\n💡 COMPARISON:")
        print("-" * 70)
        print("Character-level:")
        print("  ✅ Can generate any word (including new ones)")
        print("  ✅ Learns spelling patterns")
        print("  ❌ Slower to generate")
        print("  ❌ May make spelling errors")
        print()
        print("Word-level:")
        print("  ✅ Faster generation")
        print("  ✅ Perfect spelling (known words)")
        print("  ✅ Better long-range dependencies")
        print("  ❌ Limited to vocabulary")
        print("  ❌ Can't create new words")
        print("-" * 70)


# Create interface
interface = TextGeneratorInterface(char_model, word_model, dataset, word_dataset)

# Demo generations
interface.generate("To be or not to be", model_type='char', max_len=200, temperature=0.8, top_p=0.9)
interface.generate("Once upon a time", model_type='char', max_len=200, temperature=1.2, top_k=20)

# Compare models
interface.compare_models("The quick brown fox", max_len=200)
```

---

### Part 6: Model Evaluation and Analysis

```python
def evaluate_generation_quality(model, dataset, num_samples=100, seq_length=100):
    """
    Evaluate text generation quality.
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    # Calculate perplexity
    with torch.no_grad():
        for i in range(num_samples):
            # Random sequence
            start_idx = np.random.randint(0, len(dataset.data) - seq_length)
            input_seq = torch.tensor(dataset.data[start_idx:start_idx + seq_length]).unsqueeze(0).to(device)
            target_seq = torch.tensor(dataset.data[start_idx + 1:start_idx + seq_length + 1]).unsqueeze(0).to(device)

            # Forward
            output, _ = model(input_seq)

            # Loss
            loss = criterion(output.reshape(-1, model.vocab_size), target_seq.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_samples
    perplexity = np.exp(avg_loss)

    print(f"\n📊 Metrics:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print()
    print(f"💡 Perplexity interpretation:")
    print(f"  • Lower is better")
    print(f"  • Perplexity ≈ {perplexity:.0f} means model is '{perplexity:.0f}-ways confused' on average")
    print(f"  • Random guessing: {dataset.vocab_size}")
    print(f"  • Perfect model: 1.0")

    return perplexity


def analyze_generation_diversity(model, dataset, start_texts, num_samples=5, max_len=100):
    """
    Analyze diversity of generated texts.
    """
    print("\n" + "="*70)
    print("GENERATION DIVERSITY ANALYSIS")
    print("="*70)

    all_generations = []

    for start_text in start_texts:
        print(f"\n📝 Start: '{start_text}'")
        print("-" * 70)

        generations = []
        for i in range(num_samples):
            generated = generate_text_char(model, dataset, start_text, max_len=max_len,
                                          temperature=1.0, top_p=0.9)
            generations.append(generated)
            print(f"{i+1}. {generated[:100]}...")

        all_generations.extend(generations)

    # Calculate diversity metrics
    unique_sequences = len(set(all_generations))
    diversity_ratio = unique_sequences / len(all_generations)

    print(f"\n\n📊 Diversity Metrics:")
    print(f"  Total generations: {len(all_generations)}")
    print(f"  Unique generations: {unique_sequences}")
    print(f"  Diversity ratio: {diversity_ratio:.2%}")
    print()
    print(f"💡 Interpretation:")
    if diversity_ratio > 0.9:
        print(f"  ✅ High diversity - model generates varied outputs")
    elif diversity_ratio > 0.7:
        print(f"  ⚠️  Moderate diversity - some repetition")
    else:
        print(f"  ❌ Low diversity - model is repetitive")


# Evaluate models
char_perplexity = evaluate_generation_quality(char_model, dataset)
analyze_generation_diversity(char_model, dataset,
                             start_texts=["To be", "The quick", "Once upon"],
                             num_samples=3, max_len=100)
```

---

### Part 7: Save and Deploy

```python
def save_generator(model, dataset, filename='text_generator.pth'):
    """
    Save model and vocabulary for deployment.
    """
    print(f"\n💾 Saving model to '{filename}'...")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'level': dataset.level,
        'vocab': dataset.vocab if dataset.level == 'char' else dataset.word_to_idx,
        'model_config': {
            'embed_dim': model.embedding.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
        }
    }

    torch.save(checkpoint, filename)
    print(f"✓ Model saved!")


def load_generator(filename='text_generator.pth'):
    """
    Load saved model for inference.
    """
    print(f"\n📂 Loading model from '{filename}'...")

    checkpoint = torch.load(filename)

    # Reconstruct model
    if checkpoint['level'] == 'char':
        model = CharRNNGenerator(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['model_config']['embed_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            num_layers=checkpoint['model_config']['num_layers']
        )
    else:
        model = WordLSTMGenerator(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['model_config']['embed_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            num_layers=checkpoint['model_config']['num_layers']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded!")

    return model, checkpoint


# Save models
save_generator(char_model, dataset, 'char_generator.pth')
save_generator(word_model, word_dataset, 'word_generator.pth')

# Load and test
loaded_model, checkpoint = load_generator('char_generator.pth')
print(f"\n🎯 Testing loaded model:")
test_generation = generate_text_char(loaded_model, dataset, "To be", max_len=100, temperature=0.8)
print(test_generation)
```

---

### Part 8: Production Deployment

```python
class ProductionTextGenerator:
    """
    Production-ready text generator with error handling and caching.
    """

    def __init__(self, model_path):
        """Load model and setup."""
        self.model, self.checkpoint = load_generator(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup vocabulary
        if self.checkpoint['level'] == 'char':
            self.vocab = self.checkpoint['vocab']
            self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        else:
            self.word_to_idx = self.checkpoint['vocab']
            self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}

    def generate(self, start_text, max_len=200, temperature=1.0,
                top_k=0, top_p=0.9, seed=None):
        """
        Generate text with error handling.

        Returns:
            dict with 'text', 'success', and 'error' keys
        """
        try:
            # Set random seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            # Validate inputs
            if not start_text:
                return {'success': False, 'error': 'Empty start text'}

            if max_len < 1 or max_len > 1000:
                return {'success': False, 'error': 'max_len must be between 1 and 1000'}

            if temperature <= 0:
                return {'success': False, 'error': 'temperature must be positive'}

            # Generate
            if self.checkpoint['level'] == 'char':
                # Encode
                input_seq = torch.tensor([self.char_to_idx.get(ch, 0) for ch in start_text]).unsqueeze(0).to(self.device)

                current_text = start_text
                hidden = None

                with torch.no_grad():
                    for _ in range(max_len):
                        output, hidden = self.model(input_seq, hidden)
                        logits = output[0, -1, :] / temperature

                        # Top-k
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = torch.full_like(logits, float('-inf'))
                            logits.scatter_(0, top_k_indices, top_k_logits)

                        # Top-p
                        if top_p > 0.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            logits[indices_to_remove] = float('-inf')

                        # Sample
                        probs = F.softmax(logits, dim=-1)
                        next_idx = torch.multinomial(probs, 1).item()
                        next_char = self.idx_to_char[next_idx]
                        current_text += next_char

                        input_seq = torch.tensor([[next_idx]]).to(self.device)

                return {'success': True, 'text': current_text}

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Create production generator
prod_generator = ProductionTextGenerator('char_generator.pth')

# Test API
print("\n" + "="*70)
print("PRODUCTION API TEST")
print("="*70)

test_cases = [
    {'start_text': 'To be', 'max_len': 100, 'temperature': 0.8, 'top_p': 0.9},
    {'start_text': 'The quick', 'max_len': 150, 'temperature': 1.2, 'top_k': 20},
    {'start_text': 'Once upon a time', 'max_len': 200, 'temperature': 1.0, 'top_p': 0.95, 'seed': 42},
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test Case {i}")
    print(f"{'='*70}")
    print(f"Input: {test_case}")

    result = prod_generator.generate(**test_case)

    if result['success']:
        print(f"\n✓ Success!")
        print(f"Generated text:\n{result['text']}")
    else:
        print(f"\n❌ Error: {result['error']}")
```

---

### Project Summary

```python
print("\n" + "="*80)
print("WEEKEND PROJECT COMPLETE! 🎉")
print("="*80)

print("\n✅ WHAT YOU BUILT:")
print("-" * 80)
print("1. Character-Level RNN Generator")
print("   • LSTM-based architecture")
print("   • Generates text character-by-character")
print("   • Can create new words and spellings")
print()
print("2. Word-Level LSTM with Attention")
print("   • More sophisticated architecture")
print("   • Self-attention mechanism")
print("   • Better long-range dependencies")
print()
print("3. Multiple Sampling Strategies")
print("   • Greedy (deterministic)")
print("   • Temperature sampling")
print("   • Top-k sampling")
print("   • Nucleus (top-p) sampling")
print()
print("4. Training Pipeline")
print("   • Data preprocessing")
print("   • Training with validation")
print("   • Learning rate scheduling")
print("   • Gradient clipping")
print()
print("5. Evaluation Metrics")
print("   • Perplexity")
print("   • Diversity analysis")
print("   • Quality assessment")
print()
print("6. Production Deployment")
print("   • Model saving/loading")
print("   • Error handling")
print("   • API interface")
print("   • Reproducible generation (seeds)")
print("-" * 80)

print("\n🔑 KEY CONCEPTS MASTERED:")
print("  • Sequence-to-sequence generation")
print("  • LSTM for long-term dependencies")
print("  • Attention mechanisms")
print("  • Sampling strategies")
print("  • Model evaluation")
print("  • Production deployment")

print("\n🚀 NEXT STEPS:")
print("  1. Train on larger datasets (books, Wikipedia)")
print("  2. Implement beam search")
print("  3. Add fine-tuning capabilities")
print("  4. Create web interface")
print("  5. Experiment with GPT-style models")
print("  6. Build a chatbot!")

print("\n💡 REAL-WORLD APPLICATIONS:")
print("  • Chatbots and conversational AI")
print("  • Content generation")
print("  • Code completion")
print("  • Machine translation")
print("  • Text summarization")
print("  • Creative writing assistance")

print("\n" + "="*80)
print("🎓 Congratulations! You've built a production-ready text generator!")
print("="*80)
```

---

### Week 4 Review

```python
print("\n" + "="*80)
print("WEEK 4: RECURRENT NEURAL NETWORKS - COMPLETE!")
print("="*80)

print("\n📚 WEEK SUMMARY:")
print("-" * 80)

print("\n**Day 1: RNN Fundamentals**")
print("  • Vanilla RNN architecture")
print("  • Hidden states and memory")
print("  • Backpropagation Through Time (BPTT)")
print("  • Vanishing gradient problem")
print("  • Character-level RNN")

print("\n**Day 2: LSTM and GRU**")
print("  • Long Short-Term Memory (LSTM)")
print("  • 3 gates: Forget, Input, Output")
print("  • Gated Recurrent Unit (GRU)")
print("  • 2 gates: Reset, Update")
print("  • Text classification with LSTM")

print("\n**Day 3: Advanced Architectures**")
print("  • Bidirectional RNNs")
print("  • Sequence-to-Sequence models")
print("  • Encoder-Decoder architecture")
print("  • Teacher forcing")

print("\n**Day 4: Attention Mechanisms**")
print("  • Context vector bottleneck problem")
print("  • Bahdanau (Additive) Attention")
print("  • Luong (Multiplicative) Attention")
print("  • Self-Attention")
print("  • Seq2Seq with Attention")

print("\n**Day 5: Transformers**")
print("  • Scaled Dot-Product Attention")
print("  • Multi-Head Attention")
print("  • Positional Encoding")
print("  • Transformer architecture")
print("  • BERT vs GPT")

print("\n**Weekend Project: Text Generator**")
print("  • Character-level RNN")
print("  • Word-level LSTM with Attention")
print("  • Sampling strategies")
print("  • Production deployment")

print("\n" + "-" * 80)

print("\n🎯 SKILLS ACQUIRED:")
print("  ✅ Sequence modeling")
print("  ✅ RNN, LSTM, GRU architectures")
print("  ✅ Attention mechanisms")
print("  ✅ Transformer architecture")
print("  ✅ Text generation")
print("  ✅ Production deployment")

print("\n📈 PROGRESS:")
print("  • Week 1: Neural Networks Basics ✅")
print("  • Week 2: Training Deep Networks ✅")
print("  • Week 3: Convolutional Neural Networks ✅")
print("  • Week 4: Recurrent Neural Networks ✅")
print("  • Week 5: Coming soon...")

print("\n🚀 READY FOR WEEK 5:")
print("  → Advanced Topics:")
print("     • Generative Adversarial Networks (GANs)")
print("     • Autoencoders & VAEs")
print("     • Reinforcement Learning")
print("     • Graph Neural Networks")
print("     • Advanced Transformers")

print("\n" + "="*80)
print("🎓 WEEK 4 COMPLETE! YOU'RE NOW AN RNN EXPERT!")
print("="*80)
```

---

_End of Week 4. Total time: ~40-50 hours._

---
