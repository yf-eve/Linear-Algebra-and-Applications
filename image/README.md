# Image Processing & Restoration (2D Signals)

This module demonstrates the application of **Linear Algebra**, **Optimization**, and **Inverse Problems** on 2D image signals. Each Python script here maps a specific image processing task to a fundamental matrix operation or mathematical model.

## 🛠️ Script Overview

The scripts in this directory are structured by their mathematical approach:

| Filename | Matrix Operation / Model | Task Description |
| :--- | :--- | :--- |
| **`representation.py`** | **Basis Functions** | Exploring discrete basis transformations to understand image representation. |
| **`color.py`** | **Tensor Operations** | Color space transformations and channel manipulations. |
| **`transformation.py`**| **Affine Geometry** | **Affine transformation matrices** for image scaling, rotation, shearing, and reflection. |
| **`blur.py`** | **Convolution** | $y = A \ast x$. Low-pass filtering using a Gaussian convolution kernel to blur an image. |
| **`edge_gradient.py`** | **Differential Geometry** | Using differential operators (e.g., Sobel kernels) to compute image gradients ($G_x, G_y$). |
| **`inpainting.py`** | **Partial Differential Eq** | Discrete **Laplacian operator** and isotropic diffusion for recovering missing image regions. |
| **`deblur.py`** | **Inverse Problem** | Solving $y = Ax + v$ with **Tikhonov regularization** ($\lambda$) to restore blurred images. |




## 🔬 Core Analysis: Deblurring Inverse Problems

In `deblur.py`, we address the ill-posed nature of deconvolution using a linear regularization framework. Our analysis focuses on how the choice of the regularization parameter **$\lambda$** directly impacts the stability of the solution.

### The Regularization Trade-off

$$\hat{\mathbf{x}} = \arg \min_{\mathbf{x}} \left\{ \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{L}\mathbf{x}\|_2^2 \right\}$$

- **$\lambda \to 0$ (No Regularization)**: The solution is extremely unstable. High frequencies (noise) are amplified uncontrollably, resulting in severe "ringing" artifacts and "snow" in the output, rendering the original image indistinguishable.
- **$\lambda \to \infty$ (Strong Regularization)**: The solution is stable but **over-smoothed**. The dominant term is $\|\mathbf{Lx}\|_2^2$, which forces the solution to have very small gradients, effectively blurring the restored image back again, losing fine edges and textures.

*Choosing the optimal $\lambda$ is crucial to balancing noise amplification vs. over-smoothing.*


### Noise Amplification in Practice

A key finding during our experiments was that for **low-contrast images** (like the `woman.jpg` sample, due to a lower Signal-to-Noise Ratio), the noise amplification effect is even more pronounced. The inverse operator $A^{-1}$ treats subtle, low-contrast gradients as noise to be regularized, making the optimal $\lambda$ selection highly delicate for these cases.

## 🚀 Getting Started

1. **Install Python 3.8+**
2. **Install dependencies**: `pip install numpy scipy scikit-image matplotlib`
3. **Run deblurring**:
   ```bash
   python deblur.py