# Week 3: Convolutional Neural Networks - Complete Tutorial

## Mastering Computer Vision with CNNs

> **Expert Guide**: 15+ Years of Computer Vision Experience
>
> By the end of this week, you'll understand how CNNs work at a fundamental level, build them from scratch, and create production-quality image classifiers using modern architectures.

---

## 📚 Table of Contents

1. [Introduction: Why CNNs for Images?](#introduction)
2. [Day 1: Convolution Operation (From Scratch)](#day-1)
3. [Day 2: Building Complete CNNs](#day-2)
4. [Day 3: Famous CNN Architectures](#day-3)
5. [Day 4: Transfer Learning](#day-4)
6. [Day 5: Advanced CNN Techniques](#day-5)
7. [Weekend Project: Custom Image Classifier with Transfer Learning](#weekend-project)
8. [Week Review & Key Takeaways](#week-review)

---

<a name="introduction"></a>

## 🧠 Introduction: Why CNNs for Images?

### The Problem with Fully Connected Networks for Images

**Consider a small 32×32 RGB image:**

- Pixels: 32 × 32 × 3 = 3,072 values
- Hidden layer with 1,000 neurons = 3,072,000 parameters!
- For 224×224 images (ImageNet): 150,528 inputs → 150 million parameters in first layer!

**Problems:**

1. ❌ **Too many parameters** → overfitting, slow training
2. ❌ **No spatial awareness** → treats neighboring pixels like distant ones
3. ❌ **Not translation invariant** → cat in corner ≠ cat in center
4. ❌ **Doesn't scale** → impossible for high-resolution images

### The CNN Solution

**Key Ideas:**

1. ✅ **Local connectivity** - neurons connect to small regions
2. ✅ **Parameter sharing** - same filter used across entire image
3. ✅ **Translation invariance** - detect features anywhere
4. ✅ **Hierarchical features** - edges → shapes → objects

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compare_fc_vs_cnn():
    """
    Compare parameters: Fully Connected vs CNN.
    """
    print("="*70)
    print("FULLY CONNECTED vs CNN: PARAMETER COMPARISON")
    print("="*70)

    image_size = 224
    num_classes = 1000
    hidden_size = 4096

    # Fully Connected Network
    fc_input = image_size * image_size * 3  # 150,528
    fc_params = fc_input * hidden_size + hidden_size * num_classes

    # CNN
    # Conv1: 3 input channels, 64 filters, 3×3
    conv1_params = (3 * 3 * 3) * 64 + 64  # weights + bias
    # Conv2: 64 input channels, 128 filters, 3×3
    conv2_params = (3 * 3 * 64) * 128 + 128
    # Conv3: 128 input channels, 256 filters, 3×3
    conv3_params = (3 * 3 * 128) * 256 + 256
    # Final FC: (after pooling) 256 * 7 * 7 → 1000
    fc_final_params = (256 * 7 * 7) * num_classes + num_classes

    cnn_params = conv1_params + conv2_params + conv3_params + fc_final_params

    print(f"\n📊 PARAMETER COUNT:")
    print("-" * 70)
    print(f"  Image size: {image_size}×{image_size}×3")
    print(f"  Number of classes: {num_classes}")
    print(f"\n  Fully Connected Network:")
    print(f"    Input layer → Hidden: {fc_input:,} × {hidden_size:,} = {fc_input * hidden_size:,}")
    print(f"    Hidden → Output:      {hidden_size:,} × {num_classes:,} = {hidden_size * num_classes:,}")
    print(f"    TOTAL: {fc_params:,} parameters")

    print(f"\n  Convolutional Neural Network:")
    print(f"    Conv1 (3→64):   {conv1_params:,}")
    print(f"    Conv2 (64→128): {conv2_params:,}")
    print(f"    Conv3 (128→256): {conv3_params:,}")
    print(f"    FC Final:        {fc_final_params:,}")
    print(f"    TOTAL: {cnn_params:,} parameters")

    reduction = (1 - cnn_params / fc_params) * 100

    print(f"\n  🎯 REDUCTION: {reduction:.1f}% fewer parameters!")
    print(f"     ({fc_params / cnn_params:.1f}× smaller)")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Fully\nConnected', 'CNN']
    params = [fc_params / 1e6, cnn_params / 1e6]  # In millions
    colors = ['coral', 'skyblue']

    bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.set_title('Parameter Count: FC vs CNN for 224×224 Images', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.1f}M\nparams',
                ha='center', va='bottom', fontsize=11, weight='bold')

    # Add reduction annotation
    ax.annotate(f'{reduction:.1f}% reduction!',
                xy=(1, params[1]), xytext=(0.5, params[0] * 0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green', weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('week3_fc_vs_cnn_params.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("✓ CNNs are dramatically more efficient for images!")
    print("="*70)

compare_fc_vs_cnn()
```

### What You'll Learn This Week

**Day 1**: Convolution operation from scratch

- Mathematical foundations
- Manual implementation
- Filters and feature maps

**Day 2**: Building complete CNNs

- Pooling layers
- Complete CNN architecture
- Training pipeline

**Day 3**: Famous architectures

- LeNet, AlexNet, VGG
- ResNet, Inception
- EfficientNet

**Day 4**: Transfer learning

- Pre-trained models
- Fine-tuning strategies
- Domain adaptation

**Day 5**: Advanced techniques

- Data augmentation
- Class activation maps
- Model interpretation

**Weekend**: Custom image classifier project!

---

<a name="day-1"></a>

## 📅 Day 1: Convolution Operation (From Scratch)

> "Understanding convolution is understanding computer vision." - Yann LeCun

### 1.1 What is Convolution?

**Intuition**: Slide a small filter over an image, computing dot products.

**Mathematical Definition:**

For 2D image $I$ and filter $K$:

$$
(I * K)[i, j] = \sum_{m}\sum_{n} I[i+m, j+n] \cdot K[m, n]
$$

**Visual Example:**

```
Image (5×5):          Filter (3×3):       Output (3×3):
┌─────────────┐       ┌───────┐          ┌─────────┐
│ 1 2 3 4 5  │       │ 1 0 -1│          │ ? ? ?  │
│ 6 7 8 9 10 │   *   │ 1 0 -1│    →     │ ? ? ?  │
│11 12 13 14 15│      │ 1 0 -1│          │ ? ? ?  │
│16 17 18 19 20│      └───────┘          └─────────┘
│21 22 23 24 25│
└─────────────┘
```

### 1.2 Convolution from Scratch (NumPy)

```python
def conv2d_single_channel(image, kernel):
    """
    Perform 2D convolution on a single channel.

    Args:
        image: (H, W) array
        kernel: (K, K) array

    Returns:
        output: convolved array
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Output size (valid convolution, no padding)
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    # Slide kernel over image
    for i in range(output_h):
        for j in range(output_w):
            # Extract region
            region = image[i:i+kernel_h, j:j+kernel_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)

    return output


def demonstrate_convolution():
    """
    Demonstrate convolution operation step by step.
    """
    print("\n" + "="*70)
    print("CONVOLUTION OPERATION DEMONSTRATION")
    print("="*70)

    # Simple input image
    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=float)

    # Edge detection filter (vertical edges)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=float)

    print("\nInput Image (5×5):")
    print(image)

    print("\nKernel/Filter (3×3) - Vertical Edge Detector:")
    print(kernel)

    # Perform convolution
    output = conv2d_single_channel(image, kernel)

    print("\nOutput Feature Map (3×3):")
    print(output)

    # Show one computation step by step
    print("\n" + "="*70)
    print("STEP-BY-STEP COMPUTATION (First Position):")
    print("="*70)

    region = image[0:3, 0:3]
    print("\nRegion from image:")
    print(region)
    print("\nKernel:")
    print(kernel)
    print("\nElement-wise multiplication:")
    print(region * kernel)
    print(f"\nSum: {np.sum(region * kernel)}")
    print(f"Output[0, 0] = {output[0, 0]}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0].set_title('Input Image (5×5)', fontsize=12, weight='bold')
    axes[0].axis('off')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            axes[0].text(j, i, f'{int(image[i,j])}',
                        ha='center', va='center', color='red', fontsize=10)

    axes[1].imshow(kernel, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
    axes[1].set_title('Kernel (3×3)\nVertical Edge Detector', fontsize=12, weight='bold')
    axes[1].axis('off')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1].text(j, i, f'{int(kernel[i,j])}',
                        ha='center', va='center', color='white', fontsize=12, weight='bold')

    axes[2].imshow(output, cmap='coolwarm', interpolation='nearest')
    axes[2].set_title('Output (3×3)', fontsize=12, weight='bold')
    axes[2].axis('off')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            axes[2].text(j, i, f'{int(output[i,j])}',
                        ha='center', va='center',
                        color='white' if abs(output[i,j]) > 10 else 'black',
                        fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('week3_convolution_demo.png', dpi=150)
    plt.show()

    print("\n✓ Convolution computes local patterns using sliding window!")

demonstrate_convolution()
```

### 1.3 Common Kernels/Filters

```python
def demonstrate_common_kernels():
    """
    Show common image processing kernels.
    """
    print("\n" + "="*70)
    print("COMMON CONVOLUTION KERNELS")
    print("="*70)

    # Define kernels
    kernels = {
        'Identity': np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        'Edge Detection\n(Vertical)': np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]),
        'Edge Detection\n(Horizontal)': np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ]),
        'Sharpen': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        'Box Blur': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]) / 9.0,
        'Gaussian Blur': np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16.0,
        'Sobel X': np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]),
        'Sobel Y': np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]),
        'Emboss': np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
    }

    # Create test image
    test_image = np.zeros((50, 50))
    test_image[20:30, 15:35] = 1  # Rectangle
    test_image[10:15, 10:15] = 1  # Small square

    # Apply kernels
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Show original
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=11, weight='bold')
    axes[0].axis('off')

    for idx, (name, kernel) in enumerate(list(kernels.items())[:11], start=1):
        if idx < len(axes):
            # Apply convolution
            output = conv2d_single_channel(test_image, kernel)

            axes[idx].imshow(output, cmap='gray')
            axes[idx].set_title(name, fontsize=11, weight='bold')
            axes[idx].axis('off')

    plt.suptitle('Common Convolution Kernels and Their Effects',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('week3_common_kernels.png', dpi=150)
    plt.show()

    print("\n✓ Different kernels detect different features!")
    print("  → Edges, blurs, sharpening, etc.")
    print("  → CNNs LEARN these filters automatically!")

demonstrate_common_kernels()
```

### 1.4 Padding and Stride

**Padding**: Add zeros around image border

- **Valid**: No padding (output smaller)
- **Same**: Pad to keep same size

**Stride**: Step size when sliding filter

- Stride=1: Slide one pixel at a time
- Stride=2: Skip every other position

```python
def conv2d_with_padding_stride(image, kernel, padding=0, stride=1):
    """
    Convolution with padding and stride support.

    Args:
        image: (H, W) array
        kernel: (K, K) array
        padding: number of zeros to add around border
        stride: step size for sliding window

    Returns:
        output: convolved array
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Output size with stride
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w))

    # Slide with stride
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start+kernel_h, w_start:w_start+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output


def demonstrate_padding_stride():
    """
    Show effects of padding and stride.
    """
    print("\n" + "="*70)
    print("PADDING AND STRIDE DEMONSTRATION")
    print("="*70)

    # Test image
    image = np.random.randn(7, 7)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    # Different configurations
    configs = [
        {'padding': 0, 'stride': 1, 'name': 'Valid\n(No Padding, Stride=1)'},
        {'padding': 1, 'stride': 1, 'name': 'Same\n(Padding=1, Stride=1)'},
        {'padding': 0, 'stride': 2, 'name': 'Valid + Stride\n(No Padding, Stride=2)'},
        {'padding': 1, 'stride': 2, 'name': 'Same + Stride\n(Padding=1, Stride=2)'}
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image, cmap='coolwarm', interpolation='nearest')
    axes[0].set_title('Input Image\n(7×7)', fontsize=11, weight='bold')
    axes[0].grid(True, alpha=0.3)

    # Kernel
    axes[1].imshow(kernel, cmap='RdBu', interpolation='nearest')
    axes[1].set_title('Kernel\n(3×3)', fontsize=11, weight='bold')
    axes[1].grid(True, alpha=0.3)

    for idx, config in enumerate(configs, start=2):
        output = conv2d_with_padding_stride(
            image, kernel,
            padding=config['padding'],
            stride=config['stride']
        )

        axes[idx].imshow(output, cmap='coolwarm', interpolation='nearest')
        axes[idx].set_title(f"{config['name']}\nOutput: {output.shape[0]}×{output.shape[1]}",
                           fontsize=11, weight='bold')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Effect of Padding and Stride on Output Size',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('week3_padding_stride.png', dpi=150)
    plt.show()

    print("\n📐 OUTPUT SIZE FORMULA:")
    print("-" * 70)
    print("  Output Size = (Input + 2×Padding - Kernel) / Stride + 1")
    print("\n  Example: Input=7, Kernel=3, Padding=1, Stride=1")
    print(f"  Output = (7 + 2×1 - 3) / 1 + 1 = {(7 + 2*1 - 3) // 1 + 1}")

    print("\n✓ Padding maintains spatial dimensions!")
    print("✓ Stride reduces output size (downsampling)!")

demonstrate_padding_stride()
```

### 1.5 Multi-Channel Convolution (RGB Images)

```python
def conv2d_multi_channel(image, kernel):
    """
    Convolution for multi-channel images (e.g., RGB).

    Args:
        image: (H, W, C) array (e.g., 28×28×3)
        kernel: (K, K, C) array (e.g., 3×3×3)

    Returns:
        output: (H', W') single channel
    """
    image_h, image_w, channels = image.shape
    kernel_h, kernel_w, _ = kernel.shape

    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region across all channels
            region = image[i:i+kernel_h, j:j+kernel_w, :]
            # Sum across all channels
            output[i, j] = np.sum(region * kernel)

    return output


def demonstrate_rgb_convolution():
    """
    Show convolution on RGB images.
    """
    print("\n" + "="*70)
    print("RGB IMAGE CONVOLUTION")
    print("="*70)

    # Create simple RGB image (8×8×3)
    image = np.zeros((8, 8, 3))
    # Red square
    image[2:6, 2:6, 0] = 1.0  # Red channel
    # Green circle (approximate)
    for i in range(8):
        for j in range(8):
            if (i-4)**2 + (j-4)**2 < 9:
                image[i, j, 1] = 0.5  # Green channel

    # RGB kernel (3×3×3) - detects red edges
    kernel = np.zeros((3, 3, 3))
    kernel[:, :, 0] = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]])  # Red edge detector

    print("\nInput: RGB Image (8×8×3)")
    print("Kernel: (3×3×3) - Red vertical edge detector")

    # Apply convolution
    output = conv2d_multi_channel(image, kernel)

    print(f"Output: Single channel feature map ({output.shape[0]}×{output.shape[1]})")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # RGB image
    axes[0].imshow(image)
    axes[0].set_title('Input RGB Image\n(8×8×3)', fontsize=11, weight='bold')
    axes[0].axis('off')

    # Individual channels
    axes[1].imshow(image[:,:,0], cmap='Reds')
    axes[1].set_title('Red Channel', fontsize=11, weight='bold')
    axes[1].axis('off')

    axes[2].imshow(kernel[:,:,0], cmap='RdBu')
    axes[2].set_title('Kernel (Red Channel)\nEdge Detector', fontsize=11, weight='bold')
    axes[2].axis('off')

    axes[3].imshow(output, cmap='coolwarm')
    axes[3].set_title(f'Output Feature Map\n({output.shape[0]}×{output.shape[1]})',
                     fontsize=11, weight='bold')
    axes[3].axis('off')

    plt.suptitle('Multi-Channel (RGB) Convolution', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('week3_rgb_convolution.png', dpi=150)
    plt.show()

    print("\n✓ RGB convolution:")
    print("  → Kernel has same depth as input (3 channels)")
    print("  → Output is single channel (summed across RGB)")
    print("  → Multiple kernels → multiple output channels!")

demonstrate_rgb_convolution()
```

### 1.6 Multiple Filters → Multiple Feature Maps

```python
def conv2d_layer(image, filters, bias=None):
    """
    Convolution layer with multiple filters.

    Args:
        image: (H, W, C_in) array
        filters: (F, K, K, C_in) array - F filters
        bias: (F,) array - bias for each filter

    Returns:
        output: (H', W', F) array - F feature maps
    """
    num_filters = filters.shape[0]

    # Apply each filter
    feature_maps = []
    for f in range(num_filters):
        fmap = conv2d_multi_channel(image, filters[f])
        if bias is not None:
            fmap += bias[f]
        feature_maps.append(fmap)

    # Stack along channel dimension
    output = np.stack(feature_maps, axis=-1)

    return output


def demonstrate_multiple_filters():
    """
    Show how multiple filters create multiple feature maps.
    """
    print("\n" + "="*70)
    print("MULTIPLE FILTERS → MULTIPLE FEATURE MAPS")
    print("="*70)

    # Create test image with edges
    image = np.zeros((10, 10, 1))
    image[3:7, 2:3, 0] = 1  # Vertical edge
    image[2:3, 3:7, 0] = 1  # Horizontal edge

    # Create 3 different filters
    filters = np.zeros((3, 3, 3, 1))

    # Filter 1: Vertical edge detector
    filters[0, :, :, 0] = np.array([[1, 0, -1],
                                     [1, 0, -1],
                                     [1, 0, -1]])

    # Filter 2: Horizontal edge detector
    filters[1, :, :, 0] = np.array([[1, 1, 1],
                                     [0, 0, 0],
                                     [-1, -1, -1]])

    # Filter 3: Diagonal edge detector
    filters[2, :, :, 0] = np.array([[2, 1, 0],
                                     [1, 0, -1],
                                     [0, -1, -2]])

    # Apply all filters
    feature_maps = conv2d_layer(image, filters)

    print(f"\nInput shape: {image.shape}")
    print(f"Filters shape: {filters.shape} (3 filters, 3×3 each)")
    print(f"Output shape: {feature_maps.shape} (3 feature maps)")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Input
    axes[0, 0].imshow(image[:,:,0], cmap='gray')
    axes[0, 0].set_title('Input Image\n(10×10×1)', fontsize=10, weight='bold')
    axes[0, 0].axis('off')

    # Filters
    filter_names = ['Vertical\nEdge', 'Horizontal\nEdge', 'Diagonal\nEdge']
    for i in range(3):
        axes[0, i+1].imshow(filters[i, :, :, 0], cmap='RdBu', vmin=-2, vmax=2)
        axes[0, i+1].set_title(f'Filter {i+1}\n{filter_names[i]}', fontsize=10, weight='bold')
        axes[0, i+1].axis('off')

    # Feature maps
    axes[1, 0].axis('off')  # Empty
    for i in range(3):
        axes[1, i+1].imshow(feature_maps[:,:,i], cmap='coolwarm')
        axes[1, i+1].set_title(f'Feature Map {i+1}\n({feature_maps.shape[0]}×{feature_maps.shape[1]})',
                              fontsize=10, weight='bold')
        axes[1, i+1].axis('off')

    plt.suptitle('Multiple Filters Detect Different Features', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('week3_multiple_filters.png', dpi=150)
    plt.show()

    print("\n✓ Each filter detects different patterns!")
    print("  → Vertical edges, horizontal edges, diagonals, etc.")
    print("  → CNNs learn these filters automatically during training!")

demonstrate_multiple_filters()
```

### 1.7 Convolution in PyTorch

```python
def demonstrate_pytorch_conv():
    """
    Convolution in PyTorch vs our implementation.
    """
    print("\n" + "="*70)
    print("PYTORCH CONVOLUTION")
    print("="*70)

    # Create test data
    batch_size = 2
    in_channels = 3
    height, width = 28, 28
    out_channels = 16

    # Random input (batch, channels, height, width)
    x = torch.randn(batch_size, in_channels, height, width)

    # Convolution layer
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    )

    # Forward pass
    output = conv(x)

    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels: {in_channels}")
    print(f"  Height×Width: {height}×{width}")

    print(f"\nConv2d parameters:")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  kernel_size: 3×3")
    print(f"  stride: 1")
    print(f"  padding: 1")

    print(f"\nOutput shape: {output.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels: {out_channels}")
    print(f"  Height×Width: {output.shape[2]}×{output.shape[3]}")

    # Count parameters
    weight_params = in_channels * out_channels * 3 * 3
    bias_params = out_channels
    total_params = weight_params + bias_params

    print(f"\nParameters:")
    print(f"  Weights: {in_channels} × {out_channels} × 3 × 3 = {weight_params:,}")
    print(f"  Biases: {bias_params}")
    print(f"  Total: {total_params:,}")

    # Verify
    actual_params = sum(p.numel() for p in conv.parameters())
    print(f"  Verified: {actual_params:,} ✓")

    print("\n" + "="*70)
    print("PYTORCH CONV2D USAGE:")
    print("="*70)
    print("""
    # Create convolution layer
    conv = nn.Conv2d(
        in_channels=3,      # RGB input
        out_channels=64,    # 64 filters
        kernel_size=3,      # 3×3 kernel
        stride=1,           # Slide 1 pixel
        padding=1,          # 'Same' padding
        bias=True           # Include bias
    )

    # Forward pass
    x = torch.randn(32, 3, 224, 224)  # Batch of 32 RGB images
    output = conv(x)                   # (32, 64, 224, 224)
    """)

    print("✓ PyTorch handles all the complexity for us!")
    print("  → Efficient GPU implementation")
    print("  → Automatic gradient computation")
    print("  → Batched processing")

demonstrate_pytorch_conv()
```

### 1.8 Key Takeaways from Day 1

✅ **Convolution Operation**

- Sliding window with dot product
- Local connectivity (vs fully connected)
- Parameter sharing across image

✅ **Key Components**

- Kernels/Filters: detect patterns
- Padding: control output size
- Stride: downsample feature maps

✅ **Multi-Channel**

- RGB: 3 input channels
- Multiple filters → multiple output channels
- Each filter learns different features

✅ **Advantages**

- Fewer parameters than FC
- Translation invariant
- Hierarchical feature learning

✅ **PyTorch Usage**

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                 stride=1, padding=1)
output = conv(input)  # (B, 3, H, W) → (B, 64, H, W)
```

**Tomorrow**: Pooling + Complete CNN architectures!

---

_End of Day 1. Total time: 6-8 hours._

---

<a name="day-2"></a>

## 📅 Day 2: Building Complete CNNs

> "Pooling gives us translation invariance and reduces computation." - Geoffrey Hinton

### 2.1 Pooling Layers

**Purpose**: Downsample feature maps while retaining important information

**Types:**

1. **Max Pooling** - Take maximum value in region
2. **Average Pooling** - Take average value in region

**Benefits:**

- Reduces spatial dimensions (fewer parameters)
- Provides translation invariance
- Reduces overfitting

```python
def max_pool2d(image, pool_size=2, stride=2):
    """
    Max pooling implementation.

    Args:
        image: (H, W) or (H, W, C) array
        pool_size: size of pooling window
        stride: step size

    Returns:
        pooled array
    """
    if image.ndim == 2:
        # Single channel
        h, w = image.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1

        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                h_start = i * stride
                w_start = j * stride
                region = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
                output[i, j] = np.max(region)

        return output
    else:
        # Multi-channel
        h, w, c = image.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1

        output = np.zeros((output_h, output_w, c))

        for ch in range(c):
            output[:, :, ch] = max_pool2d(image[:, :, ch], pool_size, stride)

        return output


def average_pool2d(image, pool_size=2, stride=2):
    """Average pooling implementation."""
    if image.ndim == 2:
        h, w = image.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1

        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                h_start = i * stride
                w_start = j * stride
                region = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
                output[i, j] = np.mean(region)

        return output
    else:
        h, w, c = image.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1

        output = np.zeros((output_h, output_w, c))

        for ch in range(c):
            output[:, :, ch] = average_pool2d(image[:, :, ch], pool_size, stride)

        return output


def demonstrate_pooling():
    """
    Demonstrate max and average pooling.
    """
    print("\n" + "="*70)
    print("POOLING OPERATIONS")
    print("="*70)

    # Create test image
    image = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 3],
        [2, 1, 4, 2],
        [3, 2, 1, 5]
    ], dtype=float)

    print("\nInput Image (4×4):")
    print(image)

    # Max pooling
    max_pooled = max_pool2d(image, pool_size=2, stride=2)
    print("\nMax Pooling (2×2, stride=2):")
    print(max_pooled)

    # Average pooling
    avg_pooled = average_pool2d(image, pool_size=2, stride=2)
    print("\nAverage Pooling (2×2, stride=2):")
    print(avg_pooled)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original
    im0 = axes[0].imshow(image, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Input (4×4)', fontsize=12, weight='bold')
    axes[0].grid(True, color='white', linewidth=2)
    axes[0].set_xticks(np.arange(-0.5, 4, 1))
    axes[0].set_yticks(np.arange(-0.5, 4, 1))
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{int(image[i,j])}',
                        ha='center', va='center', color='white',
                        fontsize=14, weight='bold')

    # Max pooling
    im1 = axes[1].imshow(max_pooled, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Max Pooling (2×2)', fontsize=12, weight='bold')
    axes[1].grid(True, color='white', linewidth=2)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{int(max_pooled[i,j])}',
                        ha='center', va='center', color='white',
                        fontsize=16, weight='bold')

    # Average pooling
    im2 = axes[2].imshow(avg_pooled, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Average Pooling (2×2)', fontsize=12, weight='bold')
    axes[2].grid(True, color='white', linewidth=2)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f'{avg_pooled[i,j]:.1f}',
                        ha='center', va='center', color='white',
                        fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('week3_pooling_operations.png', dpi=150)
    plt.show()

    print("\n📊 SIZE REDUCTION:")
    print(f"  Input: 4×4 = 16 values")
    print(f"  Output: 2×2 = 4 values")
    print(f"  Reduction: 75% (4× smaller)")

    print("\n✓ Max pooling: Keeps strongest activations")
    print("✓ Average pooling: Smooth aggregation")

demonstrate_pooling()
```

### 2.2 Complete CNN Architecture

**Typical CNN Structure:**

```
Input Image
    ↓
[Conv → ReLU → Pool] × N  (Feature extraction)
    ↓
Flatten
    ↓
[FC → ReLU] × M  (Classification)
    ↓
Output
```

```python
class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.

    Architecture:
    - Conv1: 1→32 channels, 3×3
    - Pool1: MaxPool 2×2
    - Conv2: 32→64 channels, 3×3
    - Pool2: MaxPool 2×2
    - FC1: 64*7*7 → 128
    - FC2: 128 → 10
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv1 + ReLU + Pool: 28×28×1 → 14×14×32
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 + ReLU + Pool: 14×14×32 → 7×7×64
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: 7×7×64 → 3136
        x = x.view(-1, 64 * 7 * 7)

        # FC1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # FC2 (output)
        x = self.fc2(x)

        return x


def print_cnn_architecture():
    """
    Print detailed CNN architecture.
    """
    print("\n" + "="*70)
    print("CNN ARCHITECTURE BREAKDOWN")
    print("="*70)

    model = SimpleCNN()

    print("\nLayer-by-Layer Analysis:")
    print("-" * 70)

    # Input
    print("\n📥 INPUT")
    print(f"  Shape: (batch, 1, 28, 28)")
    print(f"  MNIST grayscale images")

    # Conv1
    print("\n🔵 CONV1: Conv2d(1 → 32, kernel=3×3, padding=1)")
    print(f"  Input:  (batch, 1, 28, 28)")
    print(f"  Output: (batch, 32, 28, 28)")
    print(f"  Parameters: 1×32×3×3 + 32 = {1*32*3*3 + 32:,}")

    # ReLU + Pool1
    print("\n🟢 RELU + MAXPOOL(2×2)")
    print(f"  Output: (batch, 32, 14, 14)")
    print(f"  Parameters: 0 (no learnable params)")

    # Conv2
    print("\n🔵 CONV2: Conv2d(32 → 64, kernel=3×3, padding=1)")
    print(f"  Input:  (batch, 32, 14, 14)")
    print(f"  Output: (batch, 64, 14, 14)")
    print(f"  Parameters: 32×64×3×3 + 64 = {32*64*3*3 + 64:,}")

    # ReLU + Pool2
    print("\n🟢 RELU + MAXPOOL(2×2)")
    print(f"  Output: (batch, 64, 7, 7)")

    # Flatten
    print("\n📦 FLATTEN")
    print(f"  Output: (batch, {64*7*7})")

    # FC1
    print(f"\n🔵 FC1: Linear({64*7*7} → 128)")
    print(f"  Parameters: {64*7*7}×128 + 128 = {64*7*7*128 + 128:,}")

    # Dropout
    print("\n🎲 DROPOUT(0.5)")
    print("  Parameters: 0")

    # FC2
    print("\n🔵 FC2: Linear(128 → 10)")
    print(f"  Parameters: 128×10 + 10 = {128*10 + 10:,}")

    # Total
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*70)
    print(f"📊 TOTAL PARAMETERS: {total_params:,}")
    print(f"📊 TRAINABLE PARAMETERS: {trainable_params:,}")
    print("="*70)

    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

print_cnn_architecture()
```

### 2.3 Training a CNN on MNIST

```python
def train_mnist_cnn():
    """
    Complete training pipeline for MNIST CNN.
    """
    print("\n" + "="*70)
    print("TRAINING CNN ON MNIST")
    print("="*70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    # Model
    model = SimpleCNN().to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 5
    train_losses = []
    test_losses = []
    test_accuracies = []

    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Testing
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / len(test_dataset)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 70)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, epochs + 1)

    # Loss
    axes[0].plot(epochs_range, train_losses, 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs_range, test_losses, 'r-o', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, test_accuracies, 'g-o', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([95, 100])

    plt.tight_layout()
    plt.savefig('week3_mnist_cnn_training.png', dpi=150)
    plt.show()

    print("\n" + "="*70)
    print(f"✓ TRAINING COMPLETE!")
    print(f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print("="*70)

    return model, test_loader, device

# Train the model
model, test_loader, device = train_mnist_cnn()
```

**Tomorrow:** Famous CNN architectures (LeNet, AlexNet, VGG, ResNet)!

---

_End of Day 2. Total time: 6-8 hours._

---

<a name="day-3"></a>

## 📅 Day 3: Famous CNN Architectures

> "Standing on the shoulders of giants." - Isaac Newton

### 3.1 Evolution of CNN Architectures

**Timeline of Breakthroughs:**

- 1998: **LeNet-5** - First successful CNN (digits)
- 2012: **AlexNet** - ImageNet breakthrough (deep learning era begins!)
- 2014: **VGG** - Simplicity and depth
- 2015: **ResNet** - Very deep networks (152 layers!)
- 2015: **Inception** - Multi-scale features
- 2019: **EfficientNet** - Optimal scaling

```python
def compare_architecture_milestones():
    """
    Visualize evolution of CNN architectures.
    """
    print("="*70)
    print("CNN ARCHITECTURE EVOLUTION")
    print("="*70)

    architectures = {
        'LeNet-5\n(1998)': {'params': 0.06, 'depth': 5, 'top1': 99.0, 'dataset': 'MNIST'},
        'AlexNet\n(2012)': {'params': 60, 'depth': 8, 'top1': 57.1, 'dataset': 'ImageNet'},
        'VGG-16\n(2014)': {'params': 138, 'depth': 16, 'top1': 71.5, 'dataset': 'ImageNet'},
        'ResNet-50\n(2015)': {'params': 25.6, 'depth': 50, 'top1': 76.0, 'dataset': 'ImageNet'},
        'Inception-v3\n(2015)': {'params': 23.8, 'depth': 42, 'top1': 77.5, 'dataset': 'ImageNet'},
        'EfficientNet-B7\n(2019)': {'params': 66, 'depth': 77, 'top1': 84.3, 'dataset': 'ImageNet'}
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = list(architectures.keys())

    # Parameters
    params = [architectures[name]['params'] for name in names]
    axes[0, 0].bar(range(len(names)), params, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(names))), alpha=0.7)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=0, ha='center')
    axes[0, 0].set_ylabel('Parameters (Millions)')
    axes[0, 0].set_title('Model Size (Parameters)', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Depth
    depths = [architectures[name]['depth'] for name in names]
    axes[0, 1].bar(range(len(names)), depths, color=plt.cm.Greens(np.linspace(0.4, 0.9, len(names))), alpha=0.7)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=0, ha='center')
    axes[0, 1].set_ylabel('Number of Layers')
    axes[0, 1].set_title('Network Depth', fontsize=12, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Accuracy
    accuracies = [architectures[name]['top1'] for name in names if 'ImageNet' in architectures[name]['dataset']]
    imagenet_names = [name for name in names if 'ImageNet' in architectures[name]['dataset']]
    axes[1, 0].plot(range(len(imagenet_names)), accuracies, 'ro-', linewidth=2, markersize=10)
    axes[1, 0].set_xticks(range(len(imagenet_names)))
    axes[1, 0].set_xticklabels(imagenet_names, rotation=0, ha='center')
    axes[1, 0].set_ylabel('Top-1 Accuracy (%)')
    axes[1, 0].set_title('ImageNet Accuracy Over Time', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([50, 90])

    # Timeline
    years = [1998, 2012, 2014, 2015, 2015, 2019]
    axes[1, 1].scatter(years, accuracies[:-1] + [99.0], s=200, alpha=0.6, c=range(len(names)), cmap='viridis')
    for i, (year, acc, name) in enumerate(zip(years, accuracies[:-1] + [99.0], names)):
        axes[1, 1].annotate(name.replace('\n', ' '), (year, acc),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Progress Timeline', fontsize=12, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week3_architecture_evolution.png', dpi=150)
    plt.show()

    print("\n📊 KEY TRENDS:")
    print("  1. Increasing depth: 5 → 77 layers")
    print("  2. Improving accuracy: 57% → 84% (ImageNet)")
    print("  3. More efficient: Better accuracy with fewer params")
    print("\n✓ Modern architectures are result of years of innovation!")

compare_architecture_milestones()
```

### 3.2 LeNet-5 (1998) - The Pioneer

**First successful CNN for handwritten digits**

**Architecture:**

- Conv1: 1→6, 5×5
- Pool1: AvgPool 2×2
- Conv2: 6→16, 5×5
- Pool2: AvgPool 2×2
- FC1: 120
- FC2: 84
- FC3: 10

```python
class LeNet5(nn.Module):
    """
    LeNet-5: The original CNN architecture (1998).

    Designed for 32×32 grayscale images (MNIST padded).
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv1 + Pool: 32×32×1 → 14×14×6
        x = self.pool(torch.tanh(self.conv1(x)))

        # Conv2 + Pool: 14×14×6 → 5×5×16
        x = self.pool(torch.tanh(self.conv2(x)))

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


def demonstrate_lenet():
    """
    Demonstrate LeNet-5 architecture.
    """
    print("\n" + "="*70)
    print("LeNet-5 (1998) - THE PIONEER")
    print("="*70)

    model = LeNet5()

    print("\n📋 ARCHITECTURE:")
    print("-" * 70)
    print(model)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(1, 1, 32, 32)
    output = model(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    print("\n🌟 INNOVATIONS:")
    print("  • First successful CNN architecture")
    print("  • Introduced conv → pool pattern")
    print("  • 99%+ accuracy on MNIST")
    print("  • Foundation for modern CNNs")

demonstrate_lenet()
```

### 3.3 AlexNet (2012) - The Game Changer

**Won ImageNet 2012 by huge margin (15.3% → 10.8% error)**

**Key Innovations:**

- **ReLU activation** (instead of tanh)
- **Dropout** for regularization
- **Data augmentation**
- **GPU training** (2 GPUs!)
- **Local Response Normalization**

```python
class AlexNet(nn.Module):
    """
    AlexNet: The architecture that started the deep learning revolution.

    Won ImageNet 2012 with 15.3% error (previous best: 26%).
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Conv1: 3→96, 11×11, stride=4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 96→256, 5×5
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 256→384, 3×3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 384→384, 3×3
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 384→256, 3×3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pooling (for flexible input size)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def demonstrate_alexnet():
    """
    Demonstrate AlexNet architecture.
    """
    print("\n" + "="*70)
    print("AlexNet (2012) - THE GAME CHANGER")
    print("="*70)

    model = AlexNet(num_classes=1000)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,} (~60M)")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape} (ImageNet size)")
    print(f"  Output: {output.shape} (1000 classes)")

    print("\n🌟 KEY INNOVATIONS:")
    print("  • ReLU activation (faster training)")
    print("  • Dropout (prevents overfitting)")
    print("  • Data augmentation (more training data)")
    print("  • GPU training (much faster)")
    print("  • Overlapping pooling")

    print("\n🏆 IMPACT:")
    print("  • Won ImageNet 2012")
    print("  • Started deep learning revolution")
    print("  • Showed that deeper networks work!")

demonstrate_alexnet()
```

### 3.4 VGG (2014) - Simple and Deep

**Philosophy: Deeper is better, keep it simple**

**Key Idea:**

- Only 3×3 convolutions
- Double channels after each pool
- Very deep (16-19 layers)

**VGG-16 Structure:**

```
[Conv-Conv-Pool] → 64 channels
[Conv-Conv-Pool] → 128 channels
[Conv-Conv-Conv-Pool] → 256 channels
[Conv-Conv-Conv-Pool] → 512 channels
[Conv-Conv-Conv-Pool] → 512 channels
[FC-FC-FC] → 1000 classes
```

```python
class VGG16(nn.Module):
    """
    VGG-16: Simple and deep architecture.

    Philosophy: Stack 3×3 convolutions, double channels after pooling.
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def demonstrate_vgg():
    """
    Demonstrate VGG-16 architecture.
    """
    print("\n" + "="*70)
    print("VGG-16 (2014) - SIMPLE AND DEEP")
    print("="*70)

    model = VGG16(num_classes=1000)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,} (~138M)")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    print("\n🌟 DESIGN PRINCIPLES:")
    print("  • Only 3×3 convolutions (small receptive field)")
    print("  • Stack multiple 3×3 (2× 3×3 = 5×5 receptive field)")
    print("  • Double channels after pooling")
    print("  • Very deep: 16 weight layers")

    print("\n📏 WHY 3×3 CONVOLUTIONS?")
    print("  • Two 3×3 convs = one 5×5 conv (same receptive field)")
    print("  • But fewer parameters: 2×(3×3) < 1×(5×5)")
    print("  • More non-linearity (2 ReLUs vs 1)")

    print("\n⚠️  LIMITATION:")
    print("  • Too many parameters (138M)")
    print("  • Most parameters in FC layers!")
    print("  • Slow training and inference")

demonstrate_vgg()
```

### 3.5 ResNet (2015) - Breakthrough in Deep Networks

**Problem:** Very deep networks don't train well (vanishing gradients)
**Solution:** Residual connections (skip connections)

**Key Idea:**

$$
H(x) = F(x) + x
$$

Learn residual $F(x)$ instead of direct mapping $H(x)$

```python
class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet.

    Two 3×3 convolutions with skip connection.
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

        # Skip connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add skip connection
        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet-18: 18-layer ResNet for ImageNet.
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def demonstrate_resnet():
    """
    Demonstrate ResNet architecture and advantages.
    """
    print("\n" + "="*70)
    print("ResNet (2015) - BREAKTHROUGH IN DEEP NETWORKS")
    print("="*70)

    model = ResNet18(num_classes=1000)

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,} (~11M)")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    print("\n🌟 KEY INNOVATION: RESIDUAL CONNECTIONS")
    print("  H(x) = F(x) + x")
    print("  • Learn residual F(x) = H(x) - x")
    print("  • Gradient flows directly through skip connection")
    print("  • Enables training of very deep networks (152+ layers!)")

    print("\n📊 COMPARISON:")
    print("  • VGG-16: 16 layers, 138M parameters")
    print("  • ResNet-18: 18 layers, 11M parameters (12× smaller!)")
    print("  • ResNet-50: 50 layers, 26M parameters")
    print("  • ResNet-152: 152 layers, 60M parameters")

    print("\n🏆 ACHIEVEMENTS:")
    print("  • Won ImageNet 2015")
    print("  • 3.57% error (human-level: ~5%)")
    print("  • Enabled training of 1000+ layer networks!")
    print("  • Most influential architecture in modern AI")

demonstrate_resnet()
```

### 3.6 Comparing All Architectures

```python
def comprehensive_architecture_comparison():
    """
    Side-by-side comparison of all major architectures.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE ARCHITECTURE COMPARISON")
    print("="*70)

    # Create models
    models = {
        'LeNet-5': LeNet5(),
        'AlexNet': AlexNet(),
        'VGG-16': VGG16(),
        'ResNet-18': ResNet18()
    }

    # Collect statistics
    stats = {}
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        stats[name] = {
            'params': total_params / 1e6,  # In millions
        }

    # Print table
    print("\n" + "="*70)
    print(f"{'Architecture':<15} {'Parameters':<15} {'Year':<10} {'Key Feature'}")
    print("-" * 70)
    print(f"{'LeNet-5':<15} {stats['LeNet-5']['params']:>8.2f}M      {'1998':<10} First successful CNN")
    print(f"{'AlexNet':<15} {stats['AlexNet']['params']:>8.2f}M      {'2012':<10} Deep learning era")
    print(f"{'VGG-16':<15} {stats['VGG-16']['params']:>8.2f}M     {'2014':<10} Simple & deep")
    print(f"{'ResNet-18':<15} {stats['ResNet-18']['params']:>8.2f}M      {'2015':<10} Residual connections")
    print("="*70)

    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = list(stats.keys())
    params = [stats[n]['params'] for n in names]

    # Parameters
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    bars = axes[0].bar(names, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Model Size Comparison', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M',
                    ha='center', va='bottom', fontsize=10, weight='bold')

    # Evolution timeline
    years = [1998, 2012, 2014, 2015]
    axes[1].plot(years, params, 'o-', linewidth=3, markersize=12, color='#e74c3c')
    axes[1].set_xlabel('Year', fontsize=12)
    axes[1].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[1].set_title('Architecture Evolution', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3)

    for year, param, name in zip(years, params, names):
        axes[1].annotate(name, (year, param),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Key innovations
    innovations = {
        'LeNet-5': 'Conv+Pool',
        'AlexNet': 'ReLU+Dropout',
        'VGG-16': '3×3 Convs',
        'ResNet-18': 'Skip Connections'
    }

    axes[2].axis('off')
    y_pos = 0.9
    axes[2].text(0.5, 0.95, 'Key Innovations', ha='center', fontsize=14, weight='bold', transform=axes[2].transAxes)

    for name, innovation in innovations.items():
        axes[2].text(0.1, y_pos, f'• {name}:', fontsize=11, weight='bold', transform=axes[2].transAxes)
        axes[2].text(0.4, y_pos, innovation, fontsize=11, transform=axes[2].transAxes)
        y_pos -= 0.15

    plt.tight_layout()
    plt.savefig('week3_architecture_comparison.png', dpi=150)
    plt.show()

    print("\n✓ Each architecture built upon previous innovations!")
    print("  → LeNet: Proved CNNs work")
    print("  → AlexNet: Showed deep networks scale")
    print("  → VGG: Simplified design, went deeper")
    print("  → ResNet: Enabled very deep networks")

comprehensive_architecture_comparison()
```

### 3.7 Using Pre-trained Models in PyTorch

```python
def demonstrate_pretrained_models():
    """
    Show how to use pre-trained models from torchvision.
    """
    print("\n" + "="*70)
    print("USING PRE-TRAINED MODELS")
    print("="*70)

    from torchvision import models

    # Load pre-trained ResNet-18
    print("\nLoading pre-trained ResNet-18...")
    resnet = models.resnet18(pretrained=True)

    # Model info
    total_params = sum(p.numel() for p in resnet.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")

    # Example inference
    resnet.eval()

    # Create dummy image
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = resnet(x)

    # Get top 5 predictions
    probs = F.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape} (1000 ImageNet classes)")
    print(f"\nTop 5 predictions (random input):")
    for i in range(5):
        print(f"  {i+1}. Class {top5_idx[0][i].item()}: {top5_prob[0][i].item():.4f}")

    print("\n" + "="*70)
    print("AVAILABLE PRE-TRAINED MODELS:")
    print("="*70)
    print("""
    # ResNets
    models.resnet18(pretrained=True)
    models.resnet34(pretrained=True)
    models.resnet50(pretrained=True)
    models.resnet101(pretrained=True)
    models.resnet152(pretrained=True)

    # VGG
    models.vgg16(pretrained=True)
    models.vgg19(pretrained=True)

    # Inception
    models.inception_v3(pretrained=True)

    # EfficientNet
    models.efficientnet_b0(pretrained=True)
    models.efficientnet_b7(pretrained=True)

    # MobileNet
    models.mobilenet_v2(pretrained=True)
    models.mobilenet_v3_large(pretrained=True)
    """)

    print("✓ Pre-trained models save weeks of training time!")
    print("✓ Trained on ImageNet (1.2M images, 1000 classes)")
    print("✓ Perfect for transfer learning (tomorrow's topic!)")

demonstrate_pretrained_models()
```

### 3.8 Key Takeaways from Day 3

✅ **Architecture Evolution**

- LeNet-5: First successful CNN (1998)
- AlexNet: Deep learning revolution (2012)
- VGG: Simple and deep (2014)
- ResNet: Skip connections for very deep networks (2015)

✅ **Design Principles**

- Deeper networks → better performance (until ResNet)
- Smaller filters (3×3) preferred over large (5×5, 7×7)
- Skip connections solve vanishing gradient problem
- Batch normalization stabilizes training

✅ **Modern Best Practices**

- Use ResNet-based architectures
- Start with pre-trained models
- Fine-tune for your specific task
- Balance accuracy vs efficiency

✅ **Key Innovations Summary**
| Architecture | Key Innovation | Impact |
|-------------|----------------|---------|
| LeNet-5 | Conv+Pool pattern | Proved CNNs work |
| AlexNet | ReLU, Dropout, GPUs | Started DL revolution |
| VGG | 3×3 convs, simplicity | Showed depth matters |
| ResNet | Skip connections | Enabled very deep nets |

**Tomorrow:** Transfer learning - use these powerful models for YOUR tasks!

---

_End of Day 3. Total time: 6-8 hours._

---

<a name="day-4"></a>

## 📅 Day 4: Transfer Learning

> "If I have seen further, it is by standing on the shoulders of giants." - Isaac Newton

### 4.1 What is Transfer Learning?

**Problem:** Training CNNs from scratch requires:

- Millions of images
- Weeks/months of GPU time
- Expensive hardware

**Solution:** Use pre-trained models!

```
Pre-trained on ImageNet (1.2M images) → Fine-tune on your data (1000 images)
```

**Why It Works:**

- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- Transfer low-level knowledge to new tasks

```python
def visualize_transfer_learning_concept():
    """
    Visualize what layers learn and transfer learning strategy.
    """
    print("="*70)
    print("TRANSFER LEARNING CONCEPT")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. What layers learn
    layers = ['Layer 1\n(Early)', 'Layer 2', 'Layer 3', 'Layer 4\n(Late)']
    features = [
        'Edges\nCorners\nColors',
        'Textures\nPatterns',
        'Object Parts\n(eyes, wheels)',
        'Whole Objects\n(cats, cars)'
    ]

    y_pos = np.arange(len(layers))
    axes[0, 0].barh(y_pos, [1, 2, 3, 4], color=plt.cm.Blues(np.linspace(0.3, 0.9, 4)), alpha=0.7)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels([f"{l}\n{f}" for l, f in zip(layers, features)])
    axes[0, 0].set_xlabel('Complexity →', fontsize=12)
    axes[0, 0].set_title('What CNN Layers Learn', fontsize=13, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].set_xlim([0, 5])

    # 2. Training time comparison
    scenarios = ['From\nScratch', 'Transfer\nLearning']
    times = [100, 2]  # Relative time
    colors_time = ['#e74c3c', '#2ecc71']
    bars = axes[0, 1].bar(scenarios, times, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Relative Training Time', fontsize=12)
    axes[0, 1].set_title('Training Time Comparison', fontsize=13, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for bar, time in zip(bars, times):
        height = bar.get_height()
        label = 'Weeks' if time > 50 else 'Hours'
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{label}',
                       ha='center', va='bottom', fontsize=11, weight='bold')

    # 3. Data requirement
    data_reqs = ['From\nScratch', 'Transfer\nLearning']
    data_amounts = [1000000, 1000]  # Number of images
    bars = axes[1, 0].bar(data_reqs, data_amounts, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Images Required', fontsize=12)
    axes[1, 0].set_title('Data Requirement', fontsize=13, weight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar, amount in zip(bars, data_amounts):
        height = bar.get_height()
        if amount >= 1e6:
            label = f'{amount/1e6:.1f}M'
        else:
            label = f'{amount/1e3:.0f}K'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom', fontsize=11, weight='bold')

    # 4. Transfer learning strategies
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.95, 'Transfer Learning Strategies',
                   ha='center', fontsize=14, weight='bold', transform=axes[1, 1].transAxes)

    strategies = [
        ('1. Feature Extraction', 'Freeze all layers except last', '#3498db'),
        ('2. Fine-tuning (Light)', 'Freeze early, train late layers', '#9b59b6'),
        ('3. Fine-tuning (Full)', 'Train all layers (low LR)', '#2ecc71'),
    ]

    y_pos = 0.75
    for title, desc, color in strategies:
        # Title
        axes[1, 1].text(0.1, y_pos, title, fontsize=12, weight='bold',
                       transform=axes[1, 1].transAxes, color=color)
        # Description
        axes[1, 1].text(0.15, y_pos - 0.08, desc, fontsize=10,
                       transform=axes[1, 1].transAxes, style='italic')
        y_pos -= 0.25

    plt.tight_layout()
    plt.savefig('week3_transfer_learning_concept.png', dpi=150)
    plt.show()

    print("\n🎯 KEY INSIGHT:")
    print("  Pre-trained models already know:")
    print("    • Edges and corners (Layer 1)")
    print("    • Textures and patterns (Layer 2)")
    print("    • Object parts (Layer 3)")
    print("  → You only need to teach task-specific features!")

    print("\n✓ Transfer learning = 50× faster, 100× less data!")

visualize_transfer_learning_concept()
```

### 4.2 Strategy 1: Feature Extraction

**Idea:** Use pre-trained CNN as fixed feature extractor

**Steps:**

1. Load pre-trained model
2. **Freeze** all layers (no gradient updates)
3. Replace final layer with new classifier
4. Train only the new layer

```python
def create_feature_extractor(num_classes=10):
    """
    Create a feature extractor using pre-trained ResNet.

    All layers frozen except the final classifier.
    """
    from torchvision import models

    # Load pre-trained ResNet-18
    model = models.resnet18(pretrained=True)

    print("Original model:")
    print(f"  Input: 3-channel images")
    print(f"  Output: 1000 classes (ImageNet)")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"\nModified model:")
    print(f"  Input: 3-channel images")
    print(f"  Output: {num_classes} classes (your task)")
    print(f"  Trainable params in fc: {sum(p.numel() for p in model.fc.parameters()):,}")

    # Count frozen vs trainable params
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Parameter breakdown:")
    print(f"  Frozen: {frozen_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Ratio: {trainable_params / (frozen_params + trainable_params) * 100:.2f}% trainable")

    return model


def train_feature_extractor():
    """
    Train feature extractor on CIFAR-10.
    """
    print("\n" + "="*70)
    print("FEATURE EXTRACTION TRAINING")
    print("="*70)

    # Prepare data
    from torchvision import datasets, transforms

    # ImageNet normalization (important for pre-trained models!)
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    model = create_feature_extractor(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer: Only optimize the new layer!
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    num_epochs = 5
    train_losses = []
    test_accuracies = []

    print(f"\n🏋️ Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print(f"  Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/50:.4f}")
                running_loss = 0.0

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"✓ Epoch {epoch+1}/{num_epochs}: Test Accuracy = {accuracy:.2f}%")

    print(f"\n🎉 Training complete!")
    print(f"Final accuracy: {test_accuracies[-1]:.2f}%")

    return model, test_accuracies


# Run feature extraction
model_fe, acc_fe = train_feature_extractor()
```

### 4.3 Strategy 2: Fine-Tuning

**Idea:** Unfreeze some layers and train with low learning rate

**Steps:**

1. Load pre-trained model
2. Replace final layer
3. Train final layer first (feature extraction)
4. **Unfreeze** later layers
5. Train with very low learning rate

```python
def create_finetuning_model(num_classes=10, freeze_until_layer=None):
    """
    Create model for fine-tuning.

    Args:
        num_classes: Number of output classes
        freeze_until_layer: Freeze layers up to this layer (None = train all)
    """
    from torchvision import models

    model = models.resnet18(pretrained=True)

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Freeze early layers if requested
    if freeze_until_layer is not None:
        print(f"Freezing layers up to: {freeze_until_layer}")

        # ResNet structure: conv1, bn1, layer1, layer2, layer3, layer4, fc
        layers_to_freeze = []

        if freeze_until_layer >= 1:
            layers_to_freeze.extend([model.conv1, model.bn1])
        if freeze_until_layer >= 2:
            layers_to_freeze.append(model.layer1)
        if freeze_until_layer >= 3:
            layers_to_freeze.append(model.layer2)
        if freeze_until_layer >= 4:
            layers_to_freeze.append(model.layer3)

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    # Count parameters
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Fine-tuning configuration:")
    print(f"  Frozen: {frozen_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / (frozen_params + trainable_params) * 100:.2f}%")

    return model


def train_with_finetuning():
    """
    Fine-tune pre-trained model on CIFAR-10.
    """
    print("\n" + "="*70)
    print("FINE-TUNING TRAINING")
    print("="*70)

    # Prepare data (same as before)
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Create model (freeze early layers)
    model = create_finetuning_model(num_classes=10, freeze_until_layer=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer with differential learning rates
    # Low LR for pre-trained layers, higher LR for new layer
    optimizer = torch.optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}  # New layer gets higher LR
    ])

    criterion = nn.CrossEntropyLoss()

    # Train
    num_epochs = 10
    test_accuracies = []

    print(f"\n🏋️ Fine-tuning for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print(f"  Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/50:.4f}")
                running_loss = 0.0

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"✓ Epoch {epoch+1}/{num_epochs}: Test Accuracy = {accuracy:.2f}%")

    print(f"\n🎉 Fine-tuning complete!")
    print(f"Final accuracy: {test_accuracies[-1]:.2f}%")

    return model, test_accuracies


# Run fine-tuning
model_ft, acc_ft = train_with_finetuning()
```

### 4.4 Comparing Strategies

```python
def compare_transfer_learning_strategies():
    """
    Compare different transfer learning approaches.
    """
    print("\n" + "="*70)
    print("TRANSFER LEARNING STRATEGY COMPARISON")
    print("="*70)

    # Simulate results (in practice, you'd run all experiments)
    strategies = {
        'From Scratch': {
            'accuracy': 75.0,
            'time': 100,  # relative
            'trainable_params': 11_000_000,
            'data_needed': 'Large'
        },
        'Feature Extraction': {
            'accuracy': 82.0,
            'time': 5,
            'trainable_params': 5_000,
            'data_needed': 'Small'
        },
        'Fine-tuning (Partial)': {
            'accuracy': 88.0,
            'time': 10,
            'trainable_params': 2_000_000,
            'data_needed': 'Medium'
        },
        'Fine-tuning (Full)': {
            'accuracy': 91.0,
            'time': 20,
            'trainable_params': 11_000_000,
            'data_needed': 'Medium'
        }
    }

    # Print table
    print("\n" + "="*90)
    print(f"{'Strategy':<25} {'Accuracy':<12} {'Time':<10} {'Params':<15} {'Data'}")
    print("-" * 90)

    for name, stats in strategies.items():
        params_str = f"{stats['trainable_params']/1e6:.1f}M" if stats['trainable_params'] > 1e6 else f"{stats['trainable_params']/1e3:.1f}K"
        print(f"{name:<25} {stats['accuracy']:>8.1f}%    {stats['time']:>5}×     {params_str:<15} {stats['data_needed']}")

    print("="*90)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = list(strategies.keys())
    accuracies = [strategies[n]['accuracy'] for n in names]
    times = [strategies[n]['time'] for n in names]
    params = [strategies[n]['trainable_params'] / 1e6 for n in names]

    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']

    # Accuracy
    bars = axes[0, 0].bar(range(len(names)), accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Final Accuracy', fontsize=13, weight='bold')
    axes[0, 0].set_ylim([70, 95])
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%',
                       ha='center', va='bottom', fontsize=10, weight='bold')

    # Training time
    bars = axes[0, 1].bar(range(len(names)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Relative Training Time', fontsize=12)
    axes[0, 1].set_title('Training Time', fontsize=13, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for bar, time in zip(bars, times):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{time}×',
                       ha='center', va='bottom', fontsize=10, weight='bold')

    # Trainable parameters
    bars = axes[1, 0].bar(range(len(names)), params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Trainable Parameters (Millions)', fontsize=12)
    axes[1, 0].set_title('Model Complexity', fontsize=13, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar, param in zip(bars, params):
        height = bar.get_height()
        label = f'{param:.1f}M' if param > 0.1 else f'{param*1000:.0f}K'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom', fontsize=10, weight='bold')

    # Efficiency (Accuracy / Time)
    efficiency = [acc / time for acc, time in zip(accuracies, times)]
    bars = axes[1, 1].bar(range(len(names)), efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Efficiency (Accuracy / Time)', fontsize=12)
    axes[1, 1].set_title('Training Efficiency', fontsize=13, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{eff:.2f}',
                       ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('week3_transfer_learning_comparison.png', dpi=150)
    plt.show()

    print("\n💡 RECOMMENDATIONS:")
    print("  • Small dataset (<1K images): Feature Extraction")
    print("  • Medium dataset (1K-10K): Fine-tune top layers")
    print("  • Large dataset (>10K): Fine-tune all layers")
    print("  • Very large dataset: Consider training from scratch")

    print("\n✓ Transfer learning is almost always better than training from scratch!")

compare_transfer_learning_strategies()
```

### 4.5 Domain Adaptation Techniques

**Challenge:** Pre-trained models trained on ImageNet (natural images)
**Your task:** Medical images, satellite images, X-rays, etc.

**Solution:** Domain adaptation

```python
def demonstrate_domain_adaptation():
    """
    Show techniques for adapting to different domains.
    """
    print("\n" + "="*70)
    print("DOMAIN ADAPTATION TECHNIQUES")
    print("="*70)

    print("\n📚 COMMON SCENARIOS:")
    print("-" * 70)
    print("Source Domain    → Target Domain          | Strategy")
    print("-" * 70)
    print("Natural images   → Medical images         | Fine-tune + Augmentation")
    print("Color images     → Grayscale images       | Channel adaptation")
    print("High-res images  → Low-res images         | Input preprocessing")
    print("1000 classes     → 2 classes              | Replace classifier")
    print("-" * 70)

    # Example: Adapt ResNet for grayscale medical images
    from torchvision import models

    print("\n🏥 EXAMPLE: Medical Image Classification (Grayscale)")
    print("-" * 70)

    model = models.resnet18(pretrained=True)

    print("Original model expects: 3-channel RGB images")
    print(f"First conv layer: {model.conv1}")

    # Adapt first layer for grayscale
    # Method 1: Average RGB weights
    print("\n✓ Adapting first layer for grayscale (1 channel)...")

    original_weight = model.conv1.weight.data
    print(f"  Original weight shape: {original_weight.shape}")  # [64, 3, 7, 7]

    # Average across color channels
    new_weight = original_weight.mean(dim=1, keepdim=True)
    print(f("  New weight shape: {new_weight.shape}")  # [64, 1, 7, 7]

    # Create new conv layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_weight

    print("✓ Model adapted for 1-channel input!")

    # Test
    x_gray = torch.randn(1, 1, 224, 224)
    output = model(x_gray)
    print(f"\nTest forward pass:")
    print(f"  Input: {x_gray.shape} (grayscale)")
    print(f"  Output: {output.shape}")

    print("\n" + "="*70)
    print("OTHER ADAPTATION TECHNIQUES:")
    print("="*70)
    print("""
    1. Input Preprocessing:
       • Resize to expected input size
       • Normalize using ImageNet stats
       • Convert grayscale to RGB (repeat channel)

    2. Architecture Modification:
       • Change first/last layers
       • Add domain-specific layers
       • Use multi-scale inputs

    3. Training Strategy:
       • Gradual unfreezing (freeze → partial → full)
       • Discriminative learning rates (different LR per layer)
       • Progressive resizing (small → large images)

    4. Data Augmentation:
       • Domain-specific augmentations
       • MixUp, CutMix for robustness
       • Test-time augmentation
    """)

demonstrate_domain_adaptation()
```

### 4.6 Learning Rate Strategies for Fine-Tuning

```python
def demonstrate_differential_learning_rates():
    """
    Show how to use different learning rates for different layers.
    """
    print("\n" + "="*70)
    print("DIFFERENTIAL LEARNING RATES")
    print("="*70)

    from torchvision import models

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    print("\n💡 STRATEGY:")
    print("  Early layers (general features) → Low LR (1e-5)")
    print("  Middle layers (mid-level features) → Medium LR (1e-4)")
    print("  Late layers (task-specific) → High LR (1e-3)")
    print("  New classifier layer → Highest LR (1e-2)")

    # Create optimizer with layer-wise learning rates
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters(), 'lr': 1e-5},
        {'params': model.layer1.parameters(), 'lr': 1e-5},
        {'params': model.layer2.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-3},
        {'params': model.fc.parameters(), 'lr': 1e-2}
    ])

    print("\n✓ Optimizer configured with differential learning rates!")

    # Print learning rates
    print("\nLayer-wise learning rates:")
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for i, (name, param_group) in enumerate(zip(layer_names, optimizer.param_groups)):
        print(f"  {name:10s}: LR = {param_group['lr']:.0e}")

    print("\n📊 WHY THIS WORKS:")
    print("  • Early layers: Already learned good features")
    print("  • Late layers: Need more adaptation to new task")
    print("  • New layer: Learning from scratch")

demonstrate_differential_learning_rates()
```

### 4.7 Complete Transfer Learning Pipeline

```python
class TransferLearningPipeline:
    """
    Complete pipeline for transfer learning.
    """

    def __init__(self, num_classes, architecture='resnet18', freeze_until=2):
        """
        Args:
            num_classes: Number of classes in your task
            architecture: Pre-trained architecture to use
            freeze_until: Freeze layers up to this level (0=none, 4=all)
        """
        from torchvision import models

        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model
        if architecture == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif architecture == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Replace classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        # Freeze layers
        self._freeze_layers(freeze_until)

        self.model = self.model.to(self.device)

        print(f"✓ Transfer learning pipeline initialized")
        print(f"  Architecture: {architecture}")
        print(f"  Freeze level: {freeze_until}")
        print(f"  Device: {self.device}")

    def _freeze_layers(self, freeze_until):
        """Freeze layers up to specified level."""
        if freeze_until == 0:
            return  # Train all layers

        layers_to_freeze = []

        if freeze_until >= 1:
            layers_to_freeze.extend([self.model.conv1, self.model.bn1])
        if freeze_until >= 2:
            layers_to_freeze.append(self.model.layer1)
        if freeze_until >= 3:
            layers_to_freeze.append(self.model.layer2)
        if freeze_until >= 4:
            layers_to_freeze.append(self.model.layer3)

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # Count
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Frozen params: {frozen:,}")
        print(f"  Trainable params: {trainable:,}")

    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        """Train the model."""
        # Setup optimizer with differential learning rates
        if sum(1 for p in self.model.parameters() if p.requires_grad) == self.model.fc.weight.numel() + self.model.fc.bias.numel():
            # Only classifier is trainable
            optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)
            print(f"✓ Feature extraction mode (LR={lr})")
        else:
            # Multiple layers trainable
            optimizer = torch.optim.Adam([
                {'params': self.model.layer3.parameters(), 'lr': lr/10},
                {'params': self.model.layer4.parameters(), 'lr': lr/10},
                {'params': self.model.fc.parameters(), 'lr': lr}
            ])
            print(f"✓ Fine-tuning mode (LR={lr/10} and {lr})")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

        best_val_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            val_acc = self.evaluate(val_loader)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_transfer_model.pth')

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")

        print(f"\n🎉 Training complete! Best val accuracy: {best_val_acc:.2f}%")

        return history

    def evaluate(self, data_loader):
        """Evaluate the model."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total


def demonstrate_complete_pipeline():
    """
    Demonstrate the complete transfer learning pipeline.
    """
    print("\n" + "="*70)
    print("COMPLETE TRANSFER LEARNING PIPELINE")
    print("="*70)

    # This would be your actual training code:
    print("""
# 1. Create pipeline
pipeline = TransferLearningPipeline(
    num_classes=10,
    architecture='resnet18',
    freeze_until=2  # Freeze early layers
)

# 2. Prepare your data
train_loader = ...  # Your training data
val_loader = ...    # Your validation data

# 3. Train
history = pipeline.train(
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-3
)

# 4. Evaluate
test_acc = pipeline.evaluate(test_loader)
print(f"Test accuracy: {test_acc:.2f}%")
    """)

    print("✓ This pipeline handles:")
    print("  • Loading pre-trained models")
    print("  • Freezing layers appropriately")
    print("  • Differential learning rates")
    print("  • Learning rate scheduling")
    print("  • Model checkpointing")

demonstrate_complete_pipeline()
```

### 4.8 Key Takeaways from Day 4

✅ **Transfer Learning Benefits**

- 50-100× faster training
- 10-100× less data needed
- Better generalization
- State-of-the-art results on small datasets

✅ **Three Strategies**

1. **Feature Extraction**: Freeze all, train classifier only
   - Best for: Small datasets (<1K images)
   - Fastest training
2. **Fine-Tuning (Partial)**: Freeze early, train late layers
   - Best for: Medium datasets (1K-10K images)
   - Good balance of speed and accuracy
3. **Fine-Tuning (Full)**: Train all layers with low LR
   - Best for: Large datasets (>10K images)
   - Best accuracy

✅ **Best Practices**

- Always use ImageNet normalization for pre-trained models
- Use differential learning rates (lower for early layers)
- Start with feature extraction, then fine-tune if needed
- Use data augmentation
- Monitor for overfitting (early stopping)

✅ **When to Use What**
| Dataset Size | Strategy | Expected Gain |
|-------------|----------|---------------|
| <500 images | Feature extraction | Large |
| 500-5K | Fine-tune top layers | Large |
| 5K-50K | Fine-tune all layers | Medium |
| >50K | Consider from scratch | Small |

**Tomorrow:** Advanced CNN techniques (data augmentation, visualization, interpretability)!

---

_End of Day 4. Total time: 6-8 hours._

---

<a name="day-5"></a>

## 📅 Day 5: Advanced CNN Techniques

> "In God we trust, all others must bring data." - W. Edwards Deming

### 5.1 Data Augmentation - Creating More Training Data

**Problem:** Not enough training images
**Solution:** Create variations of existing images

```python
def demonstrate_basic_augmentations():
    """
    Show basic data augmentation techniques.
    """
    print("="*70)
    print("DATA AUGMENTATION TECHNIQUES")
    print("="*70)

    from torchvision import datasets, transforms
    from PIL import Image

    # Load a sample image
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    image, label = dataset[100]  # Get one image

    # Define augmentations
    augmentations = {
        'Original': transforms.Compose([]),
        'Horizontal Flip': transforms.RandomHorizontalFlip(p=1.0),
        'Vertical Flip': transforms.RandomVerticalFlip(p=1.0),
        'Rotation 30°': transforms.RandomRotation(30),
        'Color Jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        'Random Crop': transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
        'Grayscale': transforms.Grayscale(num_output_channels=3),
        'Gaussian Blur': transforms.GaussianBlur(kernel_size=5)
    }

    # Apply and visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, (name, transform) in enumerate(augmentations.items()):
        augmented = transform(image)
        axes[idx].imshow(augmented)
        axes[idx].set_title(name, fontsize=11, weight='bold')
        axes[idx].axis('off')

    plt.suptitle('Data Augmentation Techniques', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('week3_data_augmentation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✓ Each augmentation creates a 'new' training image!")
    print("  → 1 image + 7 augmentations = 8× more data")

demonstrate_basic_augmentations()
```

### 5.2 Advanced Augmentation Strategies

```python
class AdvancedAugmentation:
    """
    Advanced augmentation techniques.
    """

    @staticmethod
    def cutout(image, n_holes=1, length=16):
        """
        Cutout: Randomly mask out square regions.

        Args:
            image: PIL Image or tensor
            n_holes: Number of cutout regions
            length: Length of cutout square
        """
        h, w = image.size(1), image.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask

        return image

    @staticmethod
    def mixup(x, y, alpha=1.0):
        """
        MixUp: Mix two images and labels.

        Args:
            x: Batch of images [N, C, H, W]
            y: Batch of labels [N]
            alpha: MixUp parameter

        Returns:
            Mixed images and labels
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    @staticmethod
    def cutmix(x, y, alpha=1.0):
        """
        CutMix: Cut and paste patches between images.

        Args:
            x: Batch of images [N, C, H, W]
            y: Batch of labels [N]
            alpha: CutMix parameter
        """
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        # Get random box
        _, _, H, W = x.shape
        cut_ratio = np.sqrt(1. - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        # Adjust lambda
        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

        return x, y, y[index], lam


def demonstrate_advanced_augmentation():
    """
    Demonstrate advanced augmentation techniques.
    """
    print("\n" + "="*70)
    print("ADVANCED AUGMENTATION: CUTOUT, MIXUP, CUTMIX")
    print("="*70)

    from torchvision import datasets, transforms

    # Load sample images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Get two images
    img1, label1 = dataset[100]
    img2, label2 = dataset[200]

    # Create batch
    images = torch.stack([img1, img2])
    labels = torch.tensor([label1, label2])

    aug = AdvancedAugmentation()

    # Apply augmentations
    cutout_img = aug.cutout(img1.clone(), n_holes=1, length=16)
    mixup_imgs, _, _, _ = aug.mixup(images.clone(), labels, alpha=1.0)
    cutmix_imgs, _, _, _ = aug.cutmix(images.clone(), labels, alpha=1.0)

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original images
    axes[0, 0].imshow(img1.permute(1, 2, 0))
    axes[0, 0].set_title('Original Image 1', fontsize=11, weight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2.permute(1, 2, 0))
    axes[0, 1].set_title('Original Image 2', fontsize=11, weight='bold')
    axes[0, 1].axis('off')

    # Cutout
    axes[0, 2].imshow(cutout_img.permute(1, 2, 0))
    axes[0, 2].set_title('Cutout (Random Erasing)', fontsize=11, weight='bold')
    axes[0, 2].axis('off')

    # Empty
    axes[0, 3].axis('off')

    # MixUp
    axes[1, 0].imshow(img1.permute(1, 2, 0))
    axes[1, 0].set_title('Image 1', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img2.permute(1, 2, 0))
    axes[1, 1].set_title('Image 2', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mixup_imgs[0].permute(1, 2, 0))
    axes[1, 2].set_title('MixUp (Blend)', fontsize=11, weight='bold')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(cutmix_imgs[0].permute(1, 2, 0))
    axes[1, 3].set_title('CutMix (Cut & Paste)', fontsize=11, weight='bold')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('week3_advanced_augmentation.png', dpi=150)
    plt.show()

    print("\n📊 AUGMENTATION COMPARISON:")
    print("-" * 70)
    print("Technique    | Description                  | When to Use")
    print("-" * 70)
    print("Cutout       | Random masking               | Improve robustness")
    print("MixUp        | Blend images & labels        | Regularization")
    print("CutMix       | Cut & paste patches          | Best of both!")
    print("-" * 70)

    print("\n✓ CutMix often gives best results (SOTA on ImageNet)!")

demonstrate_advanced_augmentation()
```

### 5.3 Grad-CAM - Visualizing What CNNs See

**Class Activation Mapping (CAM):** Show which regions the CNN focuses on

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Shows which parts of the image the CNN uses for classification.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN model
            target_layer: Layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for input image.

        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class (None = predicted class)

        Returns:
            CAM heatmap [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy(), target_class


def demonstrate_gradcam():
    """
    Demonstrate Grad-CAM visualization.
    """
    print("\n" + "="*70)
    print("GRAD-CAM: VISUALIZING CNN DECISIONS")
    print("="*70)

    from torchvision import datasets, transforms, models

    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Prepare Grad-CAM
    gradcam = GradCAM(model, model.layer4[-1])  # Last conv layer

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 sample
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    original_image, label = dataset[0]

    # Preprocess
    input_tensor = transform(original_image).unsqueeze(0)

    # Generate CAM
    cam, pred_class = gradcam.generate_cam(input_tensor)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0].axis('off')

    # Grad-CAM heatmap
    import cv2
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, weight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay (Pred: Class {pred_class})', fontsize=12, weight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('week3_gradcam.png', dpi=150)
    plt.show()

    print("\n✓ Grad-CAM shows which regions CNN uses for prediction!")
    print("  • Red = High importance")
    print("  • Blue = Low importance")

    print("\n💡 APPLICATIONS:")
    print("  • Debugging: Is model looking at right features?")
    print("  • Trust: Verify model reasoning")
    print("  • Improve: Identify dataset biases")

demonstrate_gradcam()
```

### 5.4 Feature Visualization - What Filters Learn

```python
def visualize_filters(model, layer_idx=0, num_filters=16):
    """
    Visualize what filters in a conv layer have learned.

    Args:
        model: CNN model
        layer_idx: Which conv layer to visualize
        num_filters: Number of filters to show
    """
    print("\n" + "="*70)
    print(f"VISUALIZING FILTERS (Layer {layer_idx})")
    print("="*70)

    # Get first conv layer
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if layer_idx >= len(conv_layers):
        print(f"Model only has {len(conv_layers)} conv layers!")
        return

    layer = conv_layers[layer_idx]
    filters = layer.weight.data.cpu()

    print(f"Filter shape: {filters.shape}")  # [out_channels, in_channels, H, W]

    # Visualize first num_filters
    num_filters = min(num_filters, filters.shape[0])
    rows = int(np.sqrt(num_filters))
    cols = (num_filters + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(num_filters):
        filter = filters[i]

        # If 3-channel input, show as RGB
        if filter.shape[0] == 3:
            # Normalize to [0, 1]
            filter = filter.permute(1, 2, 0)
            filter = (filter - filter.min()) / (filter.max() - filter.min())
            axes[i].imshow(filter)
        else:
            # Single channel, show as grayscale
            axes[i].imshow(filter[0], cmap='gray')

        axes[i].set_title(f'Filter {i}', fontsize=9)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Learned Filters in Layer {layer_idx}', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'week3_filters_layer{layer_idx}.png', dpi=150)
    plt.show()

    print("✓ Early layers learn edge detectors, textures")
    print("✓ Later layers learn complex patterns")


def visualize_feature_maps(model, image, layer_idx=0, num_features=16):
    """
    Visualize feature maps (activations) for a given image.

    Args:
        model: CNN model
        image: Input image tensor [1, C, H, W]
        layer_idx: Which conv layer to visualize
        num_features: Number of feature maps to show
    """
    print("\n" + "="*70)
    print(f"VISUALIZING FEATURE MAPS (Layer {layer_idx})")
    print("="*70)

    model.eval()

    # Hook to capture activations
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Register hook on target layer
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    handle = conv_layers[layer_idx].register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    handle.remove()

    # Get feature maps
    feature_maps = activations[0][0].cpu()  # [C, H, W]
    print(f"Feature map shape: {feature_maps.shape}")

    # Visualize
    num_features = min(num_features, feature_maps.shape[0])
    rows = int(np.sqrt(num_features))
    cols = (num_features + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(num_features):
        feature_map = feature_maps[i]
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature {i}', fontsize=9)
        axes[i].axis('off')

    for i in range(num_features, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Feature Maps from Layer {layer_idx}', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'week3_featuremaps_layer{layer_idx}.png', dpi=150)
    plt.show()

    print("✓ Feature maps show what patterns the network detected!")


def demonstrate_visualization():
    """
    Complete demonstration of CNN visualization.
    """
    print("\n" + "="*70)
    print("CNN VISUALIZATION COMPLETE DEMO")
    print("="*70)

    from torchvision import models, transforms, datasets

    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Visualize filters
    print("\n1. LEARNED FILTERS")
    visualize_filters(model, layer_idx=0, num_filters=16)

    # Load image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    image, _ = dataset[0]
    input_tensor = transform(image).unsqueeze(0)

    # Visualize feature maps
    print("\n2. FEATURE MAPS (ACTIVATIONS)")
    visualize_feature_maps(model, input_tensor, layer_idx=0, num_features=16)

    print("\n✓ Visualization helps understand what CNNs learn!")

demonstrate_visualization()
```

### 5.5 Model Interpretation Techniques

```python
def sensitivity_analysis(model, image, target_class, num_steps=50):
    """
    Perform sensitivity analysis to understand feature importance.

    Shows how much each pixel contributes to the prediction.
    """
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)

    model.eval()
    image.requires_grad = True

    # Get baseline prediction
    output = model(image)
    baseline_prob = F.softmax(output, dim=1)[0, target_class].item()

    print(f"Baseline probability for class {target_class}: {baseline_prob:.4f}")

    # Compute gradients
    model.zero_grad()
    output[0, target_class].backward()

    gradients = image.grad.data.abs()  # Absolute value of gradients

    # Aggregate across channels
    sensitivity = gradients.squeeze(0).mean(dim=0).cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image (denormalize for visualization)
    img_display = image.squeeze(0).detach().cpu()
    img_display = img_display * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_display = img_display.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0].axis('off')

    # Sensitivity map
    axes[1].imshow(sensitivity, cmap='hot')
    axes[1].set_title('Sensitivity Map', fontsize=12, weight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(sensitivity, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12, weight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('week3_sensitivity_analysis.png', dpi=150)
    plt.show()

    print("✓ Brighter regions = More important for prediction")

    return sensitivity


def demonstrate_interpretation():
    """
    Demonstrate various interpretation techniques.
    """
    print("\n" + "="*70)
    print("MODEL INTERPRETATION TECHNIQUES")
    print("="*70)

    print("\n📊 AVAILABLE TECHNIQUES:")
    print("-" * 70)
    print("Technique          | What it shows                    | Use case")
    print("-" * 70)
    print("Grad-CAM           | Important regions                | Classification")
    print("Sensitivity        | Pixel importance                 | Understanding decisions")
    print("Filter Viz         | What filters learn               | Debugging")
    print("Feature Maps       | Network activations              | Layer analysis")
    print("Saliency Maps      | Input gradients                  | Feature importance")
    print("-" * 70)

    print("\n💡 WHY INTERPRETATION MATTERS:")
    print("  • Debugging: Find what model learns wrong")
    print("  • Trust: Verify model reasoning")
    print("  • Improvement: Identify biases in data")
    print("  • Compliance: Explain decisions (medical, legal)")

    print("\n✓ Always visualize before deploying to production!")

demonstrate_interpretation()
```

### 5.6 Production Best Practices

```python
class ProductionCNN:
    """
    Production-ready CNN with all best practices.
    """

    def __init__(self, num_classes, architecture='resnet18'):
        """
        Initialize production CNN.
        """
        from torchvision import models

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        if architecture == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Replace classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.model = self.model.to(self.device)

        # Training transform (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Val/Test transform (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"✓ Production CNN initialized on {self.device}")

    def train(self, train_loader, val_loader, num_epochs=50, lr=1e-3):
        """
        Train with all best practices.
        """
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )

        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)

            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("✓ Training complete, best model loaded")

    def _validate(self, val_loader, criterion):
        """Validation loop."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        return val_loss, val_acc

    def predict_with_confidence(self, image):
        """
        Predict with confidence scores and top-k predictions.
        """
        self.model.eval()

        with torch.no_grad():
            # Preprocess
            if not isinstance(image, torch.Tensor):
                image = self.test_transform(image).unsqueeze(0)

            image = image.to(self.device)

            # Predict
            output = self.model(image)
            probs = F.softmax(output, dim=1)

            # Top 5
            top5_prob, top5_idx = torch.topk(probs, 5)

            return {
                'top_class': top5_idx[0][0].item(),
                'top_confidence': top5_prob[0][0].item(),
                'top5_classes': top5_idx[0].cpu().numpy(),
                'top5_confidences': top5_prob[0].cpu().numpy()
            }

    def export_onnx(self, filepath='model.onnx'):
        """
        Export model to ONNX format for deployment.
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            filepath,
            input_names=['image'],
            output_names=['output'],
            dynamic_axes={'image': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✓ Model exported to {filepath}")


def demonstrate_production_practices():
    """
    Show production best practices.
    """
    print("\n" + "="*70)
    print("PRODUCTION CNN BEST PRACTICES")
    print("="*70)

    print("\n✅ TRAINING BEST PRACTICES:")
    print("  1. Data Augmentation: Increase training data diversity")
    print("  2. Transfer Learning: Start with pre-trained weights")
    print("  3. Mixed Precision: Faster training (fp16)")
    print("  4. Gradient Clipping: Prevent exploding gradients")
    print("  5. Label Smoothing: Better calibration")
    print("  6. Weight Decay: L2 regularization")
    print("  7. Learning Rate Scheduling: OneCycle or Cosine")
    print("  8. Early Stopping: Prevent overfitting")
    print("  9. Checkpoint Saving: Save best models")
    print(" 10. Validation Set: Monitor generalization")

    print("\n✅ INFERENCE BEST PRACTICES:")
    print("  1. Test-Time Augmentation: Multiple predictions, average")
    print("  2. Model Ensembling: Combine multiple models")
    print("  3. Batch Inference: Process multiple images at once")
    print("  4. Model Quantization: INT8 for faster inference")
    print("  5. ONNX Export: Deploy anywhere")
    print("  6. TorchScript: Production deployment")

    print("\n✅ MONITORING:")
    print("  1. Log predictions and confidence scores")
    print("  2. Track accuracy over time")
    print("  3. Monitor for dataset drift")
    print("  4. A/B testing for model updates")

    print("\n✓ Follow these practices for robust production systems!")

demonstrate_production_practices()
```

### 5.7 Complete Production Example

```python
def complete_production_example():
    """
    Complete end-to-end example with all techniques.
    """
    print("\n" + "="*70)
    print("COMPLETE PRODUCTION EXAMPLE")
    print("="*70)

    print("""
# 1. Data Preparation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 2. Create Model
model = ProductionCNN(num_classes=10, architecture='resnet18')

# 3. Train with Best Practices
model.train(
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3
)

# 4. Evaluate
test_acc = model._validate(test_loader, nn.CrossEntropyLoss())
print(f"Test accuracy: {test_acc:.2f}%")

# 5. Visualize Predictions
for image, label in test_loader:
    # Predict
    result = model.predict_with_confidence(image[0])

    # Visualize with Grad-CAM
    gradcam = GradCAM(model.model, model.model.layer4[-1])
    cam, _ = gradcam.generate_cam(image[0:1].to(model.device))

    # Show results
    print(f"True: {label[0]}, Predicted: {result['top_class']}")
    print(f"Confidence: {result['top_confidence']:.2%}")

    break

# 6. Export for Deployment
model.export_onnx('production_model.onnx')

# 7. Deploy!
    """)

    print("\n✓ This pipeline includes:")
    print("  • Data augmentation")
    print("  • Transfer learning")
    print("  • Best training practices")
    print("  • Model interpretation")
    print("  • Production deployment")

complete_production_example()
```

### 5.8 Key Takeaways from Day 5

✅ **Data Augmentation**

- Basic: Flips, rotations, crops, color jitter
- Advanced: Cutout, MixUp, CutMix
- 5-10× effective data increase
- Essential for small datasets

✅ **Model Interpretation**

- **Grad-CAM**: What regions matter
- **Filter Visualization**: What filters learn
- **Feature Maps**: Network activations
- **Sensitivity Analysis**: Pixel importance
- Critical for trust and debugging

✅ **Production Best Practices**
| Component | Best Practice | Impact |
|-----------|--------------|--------|
| Training | Data augmentation | +5-10% accuracy |
| Training | Transfer learning | 50× faster |
| Training | Label smoothing | Better calibration |
| Inference | Test-time augmentation | +1-2% accuracy |
| Inference | Model ensembling | +2-3% accuracy |
| Deployment | ONNX export | Universal deployment |

✅ **Complete Pipeline**

1. Data: Augmentation + Normalization
2. Model: Transfer learning + Fine-tuning
3. Training: Best practices (LR scheduling, early stopping)
4. Validation: Grad-CAM + Metrics
5. Deployment: ONNX export + Monitoring

**Weekend Project:** Build a complete image classifier with all techniques!

---

_End of Day 5. Total time: 6-8 hours._

---

<a name="weekend-project"></a>

## 🎯 Weekend Project: Advanced Image Classifier

> "The expert in anything was once a beginner." - Helen Hayes

### Project Goal

Build a **production-ready image classifier** using everything learned this week:

- Transfer learning from pre-trained ResNet
- Advanced data augmentation (including MixUp/CutMix)
- Grad-CAM visualization
- Complete training pipeline with best practices
- Model evaluation and interpretation
- Export for deployment

**Dataset:** Custom image classification (we'll use CIFAR-100 with 100 classes)

**Target:** 70%+ accuracy with interpretable predictions

### Step 1: Project Setup and Data Preparation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

print("="*80)
print("WEEKEND PROJECT: ADVANCED IMAGE CLASSIFIER")
print("="*80)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_CLASSES = 100  # CIFAR-100

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
print("✓ Directories created")


class AdvancedTransforms:
    """
    Advanced data augmentation pipeline.
    """

    @staticmethod
    def get_train_transform():
        """Training transform with heavy augmentation."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # Cutout
        ])

    @staticmethod
    def get_test_transform():
        """Test transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# Load CIFAR-100
print("\n📦 Loading CIFAR-100 dataset...")
train_transform = AdvancedTransforms.get_train_transform()
test_transform = AdvancedTransforms.get_test_transform()

train_dataset = datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# Split train into train + validation
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"✓ Train samples: {len(train_dataset):,}")
print(f"✓ Validation samples: {len(val_dataset):,}")
print(f"✓ Test samples: {len(test_dataset):,}")

# CIFAR-100 class names
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
```

### Step 2: Model Architecture with MixUp Support

```python
class AdvancedImageClassifier(nn.Module):
    """
    Advanced image classifier with transfer learning.
    """

    def __init__(self, num_classes=100, architecture='resnet50', pretrained=True):
        super().__init__()

        print(f"\n🏗️  Building {architecture} model...")

        # Load pre-trained model
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Get number of features before classifier
        if 'resnet' in architecture:
            num_features = self.backbone.fc.in_features
            # Replace final layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        elif 'efficientnet' in architecture:
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        return self.backbone(x)


def mixup_data(x, y, alpha=1.0):
    """
    Apply MixUp augmentation.

    Args:
        x: Input images [N, C, H, W]
        y: Labels [N]
        alpha: MixUp hyperparameter

    Returns:
        Mixed inputs, targets a, targets b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Create model
model = AdvancedImageClassifier(
    num_classes=NUM_CLASSES,
    architecture='resnet50',
    pretrained=True
).to(device)

print("\n✓ Model ready for training!")
```

### Step 3: Training Loop with All Best Practices

```python
class Trainer:
    """
    Advanced trainer with all best practices.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0

        print("✓ Trainer initialized")
        print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd=1e-4)")
        print(f"  Scheduler: OneCycleLR")
        print(f"  Loss: CrossEntropy (label_smoothing=0.1)")

    def train_epoch(self, use_mixup=True, mixup_alpha=1.0):
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Apply MixUp
            if use_mixup and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

                # Forward
                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                # Normal forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model."""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs, use_mixup=True):
        """Complete training loop."""
        print(f"\n🏋️  Starting training for {num_epochs} epochs...")
        print("="*80)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            # Train
            train_loss, train_acc = self.train_epoch(use_mixup=use_mixup)

            # Validate
            val_loss, val_acc = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'checkpoints/best_model.pth')

                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                break

        print(f"\n🎉 Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Load best model
        checkpoint = torch.load('checkpoints/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Best model loaded")

        return self.history

    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=13, weight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=13, weight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, weight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting gap
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy Gap (%)', fontsize=12)
        axes[1, 1].set_title('Overfitting Gap (Train - Val)', fontsize=13, weight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/training_history.png', dpi=150)
        plt.show()

        print("✓ Training history plotted")


# Train the model
trainer = Trainer(model, train_loader, val_loader, device)
history = trainer.train(num_epochs=NUM_EPOCHS, use_mixup=True)
trainer.plot_history()
```

### Step 4: Comprehensive Evaluation

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """

    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

        self.model.eval()

    def evaluate(self):
        """Full evaluation with metrics."""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)

        all_preds = []
        all_labels = []
        all_probs = []

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f"\n📊 Overall Test Accuracy: {accuracy:.2f}%")

        # Per-class accuracy
        self._per_class_accuracy(all_preds, all_labels)

        # Top-5 accuracy
        top5_acc = self._top_k_accuracy(np.array(all_probs), np.array(all_labels), k=5)
        print(f"\n✓ Top-5 Accuracy: {top5_acc:.2f}%")

        # Confusion matrix (for top 10 classes)
        self._plot_confusion_matrix(all_preds, all_labels, top_n=10)

        return accuracy, all_preds, all_labels, all_probs

    def _per_class_accuracy(self, preds, labels, top_n=10):
        """Calculate per-class accuracy."""
        from collections import defaultdict

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for pred, label in zip(preds, labels):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        # Calculate accuracies
        class_accuracies = {
            cls: 100 * class_correct[cls] / class_total[cls]
            for cls in class_total
        }

        # Sort by accuracy
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📈 Top {top_n} Best Classes:")
        for i, (cls, acc) in enumerate(sorted_classes[:top_n], 1):
            print(f"  {i}. {self.class_names[cls]}: {acc:.2f}%")

        print(f"\n📉 Top {top_n} Worst Classes:")
        for i, (cls, acc) in enumerate(sorted_classes[-top_n:], 1):
            print(f"  {i}. {self.class_names[cls]}: {acc:.2f}%")

    def _top_k_accuracy(self, probs, labels, k=5):
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = sum([label in top_k_preds[i] for i, label in enumerate(labels)])
        return 100 * correct / len(labels)

    def _plot_confusion_matrix(self, preds, labels, top_n=10):
        """Plot confusion matrix for top N classes."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Get top N most common classes
        unique, counts = np.unique(labels, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_n:]]

        # Filter predictions and labels
        mask = np.isin(labels, top_classes)
        filtered_preds = np.array(preds)[mask]
        filtered_labels = np.array(labels)[mask]

        # Compute confusion matrix
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[self.class_names[i] for i in top_classes],
            yticklabels=[self.class_names[i] for i in top_classes],
            cbar_kws={'label': 'Accuracy'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix (Top {top_n} Classes)', fontsize=14, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=150)
        plt.show()

        print("✓ Confusion matrix plotted")


# Evaluate model
evaluator = ModelEvaluator(model, test_loader, device, cifar100_classes)
test_accuracy, all_preds, all_labels, all_probs = evaluator.evaluate()
```

### Step 5: Grad-CAM Visualization

```python
class GradCAMVisualizer:
    """
    Grad-CAM visualization for model interpretation.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM."""
        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        output[0, target_class].backward()

        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy(), target_class, F.softmax(output, dim=1)[0, target_class].item()

    def visualize_predictions(self, test_loader, class_names, num_samples=9):
        """Visualize predictions with Grad-CAM."""
        print("\n" + "="*80)
        print("GRAD-CAM VISUALIZATION")
        print("="*80)

        self.model.eval()

        # Get samples
        images, labels = next(iter(test_loader))
        images, labels = images[:num_samples], labels[:num_samples]

        fig, axes = plt.subplots(3, num_samples, figsize=(20, 9))

        import cv2

        for idx in range(num_samples):
            image = images[idx:idx+1].to(device)
            true_label = labels[idx].item()

            # Generate Grad-CAM
            cam, pred_class, confidence = self.generate_cam(image, target_class=None)

            # Denormalize image for visualization
            img_display = images[idx].cpu()
            img_display = img_display * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            # Resize CAM
            cam_resized = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))

            # Original image
            axes[0, idx].imshow(img_display)
            axes[0, idx].set_title(f'True: {class_names[true_label][:12]}', fontsize=9)
            axes[0, idx].axis('off')

            # Grad-CAM
            axes[1, idx].imshow(cam_resized, cmap='jet')
            axes[1, idx].set_title(f'Pred: {class_names[pred_class][:12]}', fontsize=9)
            axes[1, idx].axis('off')

            # Overlay
            axes[2, idx].imshow(img_display)
            axes[2, idx].imshow(cam_resized, cmap='jet', alpha=0.5)
            axes[2, idx].set_title(f'Conf: {confidence:.2%}', fontsize=9)
            axes[2, idx].axis('off')

        # Row labels
        axes[0, 0].set_ylabel('Original', fontsize=12, weight='bold')
        axes[1, 0].set_ylabel('Grad-CAM', fontsize=12, weight='bold')
        axes[2, 0].set_ylabel('Overlay', fontsize=12, weight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/gradcam_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("✓ Grad-CAM visualizations created")


# Visualize with Grad-CAM
# Get last conv layer
if hasattr(model.backbone, 'layer4'):
    target_layer = model.backbone.layer4[-1]
else:
    target_layer = list(model.backbone.children())[-2]

gradcam_viz = GradCAMVisualizer(model, target_layer)
gradcam_viz.visualize_predictions(test_loader, cifar100_classes, num_samples=9)
```

### Step 6: Model Export and Deployment

```python
def export_model_for_deployment(model, save_path='checkpoints'):
    """
    Export model in multiple formats for deployment.
    """
    print("\n" + "="*80)
    print("EXPORTING MODEL FOR DEPLOYMENT")
    print("="*80)

    model.eval()

    # 1. PyTorch state dict
    torch.save(model.state_dict(), f'{save_path}/model_state_dict.pth')
    print("✓ PyTorch state dict saved")

    # 2. Complete model
    torch.save(model, f'{save_path}/complete_model.pth')
    print("✓ Complete model saved")

    # 3. TorchScript
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(f'{save_path}/model_torchscript.pt')
    print("✓ TorchScript model saved")

    # 4. ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f'{save_path}/model.onnx',
        input_names=['image'],
        output_names=['output'],
        dynamic_axes={'image': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print("✓ ONNX model saved")

    print("\n📦 Model exported in multiple formats:")
    print(f"  • PyTorch: {save_path}/model_state_dict.pth")
    print(f"  • TorchScript: {save_path}/model_torchscript.pt")
    print(f"  • ONNX: {save_path}/model.onnx")

    # Model size
    import os
    size_mb = os.path.getsize(f'{save_path}/model_state_dict.pth') / (1024 * 1024)
    print(f"\n💾 Model size: {size_mb:.2f} MB")


export_model_for_deployment(model)
```

### Step 7: Final Summary and Report

```python
def generate_project_report(test_accuracy, history):
    """
    Generate comprehensive project report.
    """
    print("\n" + "="*80)
    print("PROJECT SUMMARY REPORT")
    print("="*80)

    print("\n🎯 PROJECT GOAL: Advanced Image Classifier on CIFAR-100")
    print("="*80)

    print("\n📊 RESULTS:")
    print(f"  • Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"  • Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  • Training Epochs: {len(history['train_loss'])}")

    print("\n✅ TECHNIQUES IMPLEMENTED:")
    print("  1. Transfer Learning (Pre-trained ResNet-50)")
    print("  2. Advanced Data Augmentation:")
    print("     • Random crops, flips, rotations")
    print("     • Color jittering")
    print("     • Random erasing (Cutout)")
    print("     • MixUp during training")
    print("  3. Training Best Practices:")
    print("     • AdamW optimizer with weight decay")
    print("     • OneCycleLR scheduling")
    print("     • Label smoothing (0.1)")
    print("     • Gradient clipping")
    print("     • Early stopping")
    print("  4. Model Interpretation:")
    print("     • Grad-CAM visualization")
    print("     • Per-class accuracy analysis")
    print("     • Confusion matrix")
    print("  5. Deployment:")
    print("     • PyTorch, TorchScript, ONNX formats")

    print("\n📈 PERFORMANCE BREAKDOWN:")
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    overfitting_gap = final_train_acc - final_val_acc

    print(f"  • Train Accuracy: {final_train_acc:.2f}%")
    print(f"  • Validation Accuracy: {final_val_acc:.2f}%")
    print(f"  • Test Accuracy: {test_accuracy:.2f}%")
    print(f"  • Overfitting Gap: {overfitting_gap:.2f}%")

    if overfitting_gap < 5:
        print("  ✓ Model generalizes well!")
    else:
        print("  ⚠️  Some overfitting detected")

    print("\n🎓 SKILLS DEMONSTRATED:")
    print("  ✓ Deep CNN architectures")
    print("  ✓ Transfer learning")
    print("  ✓ Data augmentation strategies")
    print("  ✓ Training optimization techniques")
    print("  ✓ Model interpretation and visualization")
    print("  ✓ Production deployment")

    print("\n🚀 NEXT STEPS:")
    print("  1. Try different architectures (EfficientNet, Vision Transformer)")
    print("  2. Implement test-time augmentation")
    print("  3. Build model ensemble")
    print("  4. Deploy to production (Flask API, Docker)")
    print("  5. Monitor performance in production")

    print("\n" + "="*80)
    print("🎉 WEEKEND PROJECT COMPLETE!")
    print("="*80)

    # Save report to file
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(f"# Advanced Image Classifier - Project Report\n\n")
        f.write(f"## Results\n")
        f.write(f"- **Test Accuracy**: {test_accuracy:.2f}%\n")
        f.write(f"- **Best Val Accuracy**: {max(history['val_acc']):.2f}%\n")
        f.write(f"- **Training Epochs**: {len(history['train_loss'])}\n\n")
        f.write(f"## Techniques Implemented\n")
        f.write(f"1. Transfer Learning (ResNet-50)\n")
        f.write(f"2. Advanced Data Augmentation\n")
        f.write(f"3. Training Best Practices\n")
        f.write(f"4. Model Interpretation (Grad-CAM)\n")
        f.write(f"5. Multi-format Deployment\n\n")
        f.write(f"## Files Generated\n")
        f.write(f"- `checkpoints/best_model.pth`: Best model checkpoint\n")
        f.write(f"- `checkpoints/model.onnx`: ONNX export\n")
        f.write(f"- `visualizations/training_history.png`: Training plots\n")
        f.write(f"- `visualizations/gradcam_predictions.png`: Grad-CAM visualizations\n")
        f.write(f"- `visualizations/confusion_matrix.png`: Confusion matrix\n")

    print("\n✓ Report saved to PROJECT_REPORT.md")


generate_project_report(test_accuracy, history)
```

### Weekend Project Summary

**What You Built:**

- ✅ Production-ready image classifier for 100 classes
- ✅ 70%+ accuracy on CIFAR-100 (challenging dataset!)
- ✅ Complete training pipeline with all best practices
- ✅ Model interpretation with Grad-CAM
- ✅ Multi-format deployment (PyTorch, ONNX, TorchScript)

**Key Achievements:**

1. **Transfer Learning**: Leveraged pre-trained ResNet-50
2. **Advanced Augmentation**: MixUp, Cutout, color jittering
3. **Optimal Training**: OneCycleLR, label smoothing, early stopping
4. **Interpretability**: Grad-CAM shows what model sees
5. **Production-Ready**: Exported for real-world deployment

**Files Created:**

- `checkpoints/best_model.pth` - Best model weights
- `checkpoints/model.onnx` - ONNX deployment format
- `visualizations/training_history.png` - Training curves
- `visualizations/gradcam_predictions.png` - Model interpretations
- `visualizations/confusion_matrix.png` - Class performance
- `PROJECT_REPORT.md` - Comprehensive report

**Time Spent:** 8-10 hours (including training time)

---

_End of Weekend Project._

---

<a name="week-review"></a>

## 🎓 Week 3 Complete Review

### What You Learned This Week

**Day 1: Convolution Operation**

- ✅ Manual convolution implementation
- ✅ Filters for edge detection, blur, sharpen
- ✅ Padding and stride
- ✅ Multi-channel convolution

**Day 2: Building CNNs**

- ✅ Pooling layers (max, average)
- ✅ Complete CNN architecture
- ✅ Training pipeline
- ✅ MNIST classification

**Day 3: Famous Architectures**

- ✅ LeNet-5 (1998)
- ✅ AlexNet (2012)
- ✅ VGG-16 (2014)
- ✅ ResNet (2015)
- ✅ Architecture evolution

**Day 4: Transfer Learning**

- ✅ Feature extraction
- ✅ Fine-tuning strategies
- ✅ Domain adaptation
- ✅ Differential learning rates

**Day 5: Advanced Techniques**

- ✅ Data augmentation (MixUp, CutMix)
- ✅ Grad-CAM visualization
- ✅ Model interpretation
- ✅ Production best practices

**Weekend: Complete Classifier**

- ✅ 70%+ accuracy on CIFAR-100
- ✅ All techniques combined
- ✅ Production deployment
- ✅ Comprehensive evaluation

### Key Concepts Mastered

| Concept           | Why It Matters                |
| ----------------- | ----------------------------- |
| Convolution       | Foundation of computer vision |
| Transfer Learning | 50× faster, 100× less data    |
| Data Augmentation | Improve generalization        |
| Grad-CAM          | Understand model decisions    |
| ResNet            | State-of-the-art architecture |

### Your Progress

**Complexity Growth:**

```
Week 1: Single Neuron → Multilayer Network
Week 2: Training Techniques → ResNet
Week 3: CNN Fundamentals → Production Classifier
```

**Projects Completed:**

1. MNIST Digit Classifier (98%+ accuracy)
2. CIFAR-10 with ResNet (85%+ accuracy)
3. CIFAR-100 Advanced Classifier (70%+ accuracy)

### Next Week Preview: Recurrent Neural Networks

**Topics:**

- Sequence modeling
- LSTM and GRU
- Bidirectional RNNs
- Attention mechanisms
- Text generation
- **Weekend:** Build a chatbot!

### Congratulations! 🎉

You've mastered **Convolutional Neural Networks**—the foundation of computer vision!

You can now:

- ✅ Build CNNs from scratch
- ✅ Use transfer learning effectively
- ✅ Interpret model decisions
- ✅ Deploy production models
- ✅ Achieve state-of-the-art results

**Keep building, keep learning!** 🚀

---

_Week 3 Complete. Total time: ~50 hours across 7 days._

---
