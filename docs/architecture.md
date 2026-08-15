# FastNAF-SR V5 Architecture

This document provides a comprehensive technical overview of the **FastNAF-SR V5** architecture designed for the KLA Semiconductor AI Image Restoration challenge.

---

## 1. Architectural Overview

Semiconductor inspection imaging involves complex composite degradations:
- Multiplicative / Poisson speckle noise from coherent illumination
- Gaussian thermal and electronic sensor noise
- Optical blur and 2× spatial resolution reduction
- High-frequency edge preservation constraints under ultra-low latency

To address these challenges under strict memory and throughput budgets, **FastNAF-SR V5** leverages Nonlinear Activation Free (NAF) blocks with an asymmetric Low-Resolution restoration and High-Resolution refinement pipeline.

```
                    ┌─────────────────────────┐
                    │ Input: 1 × 128 × 128    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ 3×3 Conv (1 → 48)       │
                    │ Shallow Feature Extractor│
                    └───────────┬─────────────┘
                                │
                     ┌──────────┴──────────┐
                     │ (Residual Skip)     │
                     ▼                     │
        ┌─────────────────────────┐        │
        │ 8× NAF Blocks (C = 48)  │        │
        │ Low-Res Feature Restore │        │
        └────────────┬────────────┘        │
                     │                     │
                     ▼                     │
        ┌─────────────────────────┐        │
        │ Residual Feature Fusion ├────────┘
        │ (Element-wise Addition) │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ 3×3 Conv (48 → 192)     │
        │ PixelShuffle (×2)       │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ 4× NAF Blocks (C = 48)  │
        │ High-Res Refinement     │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ 3×3 Conv (48 → 1)       │
        │ Output Reconstruction   │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ Output: 1 × 256 × 256   │
        └─────────────────────────┘
```

---

## 2. Component Details

### 2.1 Shallow Feature Extraction
- **Input shape:** `(B, 1, 128, 128)`
- **Layer:** Single 3×3 Convolution mapping 1 input channel to 48 intermediate feature channels.
- **Output:** `(B, 48, 128, 128)`

### 2.2 Low-Resolution Restoration Trunk (8 NAF Blocks)
The core denoising and deblurring operations occur in the low-resolution latent space (`128 × 128`), minimizing computational complexity ($O(HW)$) before upscaling.

Each **NAF Block** consists of:
1. **LayerNorm2d:** Channel-wise spatial normalization.
2. **1×1 Conv:** Expands channels from $C$ to $2C$ ($48 \to 96$).
3. **3×3 Depthwise Conv:** Spatial feature mixing with group count equal to channels ($96 \to 96$).
4. **SimpleGate (SG):** Replaces conventional non-linear activations (GELU/ReLU) with an element-wise product of channel splits:
   $$\text{SimpleGate}(X) = X_1 \odot X_2, \quad \text{where } X_1, X_2 \in \mathbb{R}^{B \times C \times H \times W}$$
5. **Simplified Channel Attention (SCA):**
   $$\text{SCA}(X) = X \odot \text{Conv}_{1\times 1}(\text{GlobalAvgPool}(X))$$
6. **1×1 Projection Conv:** Maps features back to $C = 48$.
7. **Feed-Forward Network (FFN):** LayerNorm2d $\to$ 1×1 Conv ($48 \to 96$) $\to$ SimpleGate $\to$ 1×1 Conv ($48 \to 48$).
8. **Residual Connections with learnable scale parameters ($\beta$).**

### 2.3 Residual Feature Fusion
The initial shallow features are added back to the restored low-resolution features:
$$F_{\text{fused}} = F_{\text{shallow}} + F_{\text{lr\_restored}}$$
This preserves low-frequency semiconductor structural patterns and stabilizes deep gradient backpropagation.

### 2.4 PixelShuffle ×2 Upsampling
- A 3×3 convolution expands channels from $48 \to 192$ ($48 \times 2^2$).
- **PixelShuffle(2)** rearranges the spatial dimensions to produce $(B, 48, 256, 256)$ without introducing checkerboard artifacts common in transposed convolutions.

### 2.5 High-Resolution Refinement Trunk (4 NAF Blocks)
4 NAF blocks operate at the full $256 \times 256$ spatial resolution to refine edge sharpness, eliminate artifacting along sub-micron IC circuit lines, and restore fine semiconductor textures.

### 2.6 Reconstruction Head
A final 3×3 convolution maps the 48 feature channels back to 1 grayscale channel, outputting the restored $256 \times 256$ image.

---

## 3. Loss Formulation

The network is trained using a multi-term hybrid objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{recon}} \mathcal{L}_{\text{Charbonnier}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{edge}} \mathcal{L}_{\text{Edge}}$$

1. **Charbonnier Loss ($\mathcal{L}_{\text{Charbonnier}}$):**
   $$\mathcal{L}_{\text{Charbonnier}}(Y, \hat{Y}) = \frac{1}{N}\sum_{i=1}^N \sqrt{(Y_i - \hat{Y}_i)^2 + \epsilon^2}, \quad \epsilon = 10^{-6}$$
   Provides robust handling of outliers and gradient stability near zero.

2. **Differentiable SSIM Loss ($\mathcal{L}_{\text{SSIM}}$):**
   $$\mathcal{L}_{\text{SSIM}}(Y, \hat{Y}) = 1.0 - \text{SSIM}(Y, \hat{Y})$$
   Enforces structural similarity and perceptual alignment.

3. **Sobel Edge Loss ($\mathcal{L}_{\text{Edge}}$):**
   $$\mathcal{L}_{\text{Edge}}(Y, \hat{Y}) = \|\nabla_x Y - \nabla_x \hat{Y}\|_1 + \|\nabla_y Y - \nabla_y \hat{Y}\|_1$$
   Penalizes edge blurring and ensures crisp IC boundary delineation.

---

## 4. Key Specifications

| Parameter | Value |
|:---|:---|
| **Model Parameters** | ~378,000 (~0.38M) |
| **Model Checkpoint Size** | 2.5 MB (.pth) / 1.5 MB (traced .pt) |
| **Input Dimensions** | 1 × 128 × 128 Grayscale |
| **Output Dimensions** | 1 × 256 × 256 Grayscale |
| **Upscaling Factor** | 2× Super-Resolution |
| **Inference Latency** | ~2.5 ms on GPU / ~45 ms on CPU |
