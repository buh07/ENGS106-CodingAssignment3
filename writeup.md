# Coding Assignment 3 — Write-Up
## Benjamin Huh
## ENGS 106 26W, Professor Chin

## Part 3A: Sparse Autoencoder

### Model Description

The sparse autoencoder is a two-layer neural network trained to reconstruct its input while learning a compressed, sparse representation in the hidden layer. The architecture consists of:

- **Input layer**: 64 units (flattened 8x8 image patches)
- **Hidden layer**: 25 units with sigmoid activation
- **Output layer**: 64 units with sigmoid activation

The cost function combines three terms:

1. **Squared error**: $\frac{1}{2m}\|a^{(3)} - x\|^2$ — reconstruction loss
2. **Weight decay**: $\frac{\lambda}{2}(\|W_1\|^2 + \|W_2\|^2)$ with $\lambda = 0.0001$ — L2 regularization to prevent overfitting
3. **KL sparsity penalty**: $\beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ with $\rho = 0.01$, $\beta = 3$ — encourages hidden units to have low average activation, promoting sparse representations

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Patch size | 8x8 |
| Number of patches | 10,000 |
| Hidden units | 25 |
| Sparsity target ($\rho$) | 0.01 |
| Weight decay ($\lambda$) | 0.0001 |
| Sparsity weight ($\beta$) | 3 |
| Optimizer | L-BFGS (2000 iterations) |

These are the default hyperparameters provided in the assignment. The sparsity target of 0.01 forces each hidden unit to activate for roughly 1% of inputs, ensuring the network learns localized features. The weight decay prevents any single weight from dominating, and beta=3 provides sufficient penalty to enforce sparsity without overwhelming the reconstruction objective.

### Evaluation

- **Untrained cost**: 48.85
- **Trained cost**: 0.461
- **L1 reconstruction error**: 7.41

The trained filters (visualized via `displayNetwork(W1.T)`) show Gabor-like edge detectors at various orientations, which is the expected result for sparse autoencoders trained on natural image patches. The reconstruction error confirms the network faithfully reproduces input patches while maintaining sparse hidden activations.

The gradient check was validated using centered differences ($\varepsilon = 10^{-4}$) on a reduced network (10 samples, 5 hidden units), yielding a relative difference of approximately $5 \times 10^{-11}$, confirming correct backpropagation implementation.

### Reflections

Sparse autoencoders demonstrate how imposing constraints on learned representations can yield meaningful, interpretable features. The emergence of edge-detector-like filters from unsupervised learning on natural images mirrors findings in neuroscience about V1 simple cells, highlighting how sparsity as an inductive bias can lead to biologically plausible representations.

## Part 3B: Transformer Language Model

### Model Description

The model is a decoder-only Transformer trained on character-level Shakespeare text. The architecture follows the "Attention Is All You Need" framework with the following components:

- **Token embedding**: Maps each character to a 384-dimensional vector
- **Positional encoding**: Sinusoidal embeddings encoding position information using $PE(pos, 2i) = \sin(pos / 10000^{2i/d})$ and $PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})$
- **Transformer blocks** (x6): Each containing:
  - **Multi-head causal self-attention** (6 heads, 64-dim each): Computes $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d_k})V$ with a causal mask to prevent attending to future tokens
  - **Feed-forward network**: Two linear layers with ReLU activation and 4x expansion (384 → 1536 → 384)
  - **Layer normalization** and **residual connections**
- **Output projection**: Linear layer mapping embeddings back to vocabulary logits

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Context length | 256 tokens |
| Embedding dimension | 384 |
| Number of layers | 6 |
| Number of attention heads | 6 |
| Attention dimension | 64 |
| Dropout | 0.2 |
| Total parameters | 10,690,625 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Training iterations | 2000 |

The model parameters were kept at the provided defaults, which balance expressiveness with trainability on CPU. The 6-layer, 6-head configuration provides sufficient depth for learning character-level patterns in Shakespeare while keeping the model manageable (~10.7M parameters).

### Evaluation

**Training Results:**

| Metric | Value |
|--------|-------|
| Final training loss | 1.155 |
| Final validation loss | 0.988 |
| Training time | 6 hours 32 minutes (CPU) |

The validation loss (0.988) being lower than training loss (1.155) is expected here due to dropout being active during training but disabled during evaluation, effectively giving the evaluation model more capacity.

**Generated Sample (prompt: "ROMEO:"):**

```
ROMEO:
Many is what loss to do it.

BENVOLIO:
I would he were it were any live a man in it
doing in this lamentation.

BRUTUS:
I wot ruled for the ground of gracious lady,
I would not say the new to: I can could not know
The renowned to death to be a trait
```

The generated text demonstrates that the model has learned:
- Character-level English spelling and word formation
- Shakespeare's dramatic structure (character names followed by dialogue)
- Iambic-like rhythm and archaic vocabulary ("wot," "renowned")
- Coherent sentence fragments, though not fully semantically coherent

### Reflections

Building a Transformer from scratch reinforces how attention mechanisms enable each token to dynamically weight information from all preceding tokens, rather than relying on fixed-size context windows. The causal masking is crucial for autoregressive generation, and the multi-head design allows the model to attend to different aspects of the input simultaneously. Even with a relatively small model trained on CPU, the results are impressively Shakespeare-like, demonstrating the power of the Transformer architecture.

## Part 3C: Denoising Diffusion Probabilistic Model (DDPM)

### Model Description

The DDPM generates handwritten digit images (MNIST) by learning to reverse a gradual noising process. The model consists of three main components:

1. **Noise Scheduler**: Defines a linear schedule of noise levels $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$ timesteps. Key quantities include cumulative products $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ that allow direct sampling of any noised version: $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon$

2. **U-Net Architecture**: A convolutional encoder-decoder with skip connections that predicts the noise $\varepsilon_\theta(x_t, t)$ added to an image. It incorporates:
   - Sinusoidal timestep embeddings (identical formulation to Transformer positional encoding)
   - Encoder path: ConvBlocks with downsampling (MaxPool2d)
   - Decoder path: ConvBlocks with upsampling + concatenated skip connections
   - Time embedding injection via learned linear projection added to intermediate features

3. **Sampling**: Reverse diffusion generates images from pure noise by iteratively denoising: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t,t)\right) + \sigma_t z$

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Image size | 28x28 (MNIST) |
| Channels | 1 (grayscale) |
| Hidden dimensions | 64 |
| Diffusion timesteps (T) | 1000 |
| Beta schedule | Linear, 1e-4 to 0.02 |
| Total parameters | 1,011,969 |
| Batch size | 128 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| Epochs | 10 |

### Evaluation

**Training Loss per Epoch:**

| Epoch | Loss |
|-------|------|
| 1 | 0.0766 |
| 2 | 0.0373 |
| 3 | 0.0332 |
| 4 | 0.0308 |
| 5 | 0.0291 |
| 6 | 0.0280 |
| 7 | 0.0272 |
| 8 | 0.0268 |
| 9 | 0.0259 |
| 10 | 0.0258 |

The training loss (MSE between predicted and actual noise) decreases steadily across epochs, with the steepest improvement in the first 2 epochs and gradual convergence thereafter. The final loss of 0.0258 indicates the model accurately predicts the noise component at each diffusion step.

The generated digit samples and the reverse diffusion trajectory visualization (showing gradual denoising from t=1000 to t=0) are included in the notebook. The generated images show recognizable digit-like structures after 10 epochs of training.

### Reflections

Diffusion models offer a fundamentally different approach to generation compared to GANs or VAEs: rather than learning a direct mapping from latent space to data, they learn to iteratively refine noise into data. The forward process is fixed and parameter-free, while all learning happens in the reverse denoising network. The connection between sinusoidal timestep embeddings here and positional encodings in Transformers highlights how the same mathematical tools recur across different deep learning paradigms. The 1000-step sampling process is slow but produces high-quality results, explaining the ongoing research into faster sampling methods like DDIM.
