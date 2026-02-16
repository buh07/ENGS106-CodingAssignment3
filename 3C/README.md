# Denoising Diffusion Probabilistic Models (DDPM)

This demo showcases training a **Denoising Diffusion Probabilistic Model (DDPM)** for image generation.

Inspired by the seminal paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239), this implementation is designed to be simple and easy to understand, making it an excellent starting point for beginners learning how to build generative models from scratch using PyTorch.

The demo is trained on the MNIST dataset (28x28 grayscale handwritten digits). The model can generate new digit images from random noise after training for approximately 10 epochs.

## Install and Start

1. Install dependencies.

You need the following Python packages:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

Or if using conda:
```bash
conda install pytorch torchvision matplotlib numpy tqdm -c pytorch
```

2. Open and run `assignment3C.ipynb`.

The model will start training on the MNIST dataset. Training loss will be printed after each epoch.

## What You Will Implement

In this assignment, you will implement:

1. **Noise Scheduler** (`NoiseScheduler.__init__`): Define the linear noise schedule for the forward diffusion process.

2. **Sinusoidal Position Embedding** (`SinusoidalPositionEmbedding.forward`): Encode timesteps using sinusoidal embeddings (similar to Transformers).

3. **U-Net Forward Pass** (`SimpleUNet.forward`): Implement the encoder-decoder architecture with skip connections.

4. **Training Loop** (`train_one_epoch`): Sample timesteps, add noise, predict noise, and compute loss.

5. **Sampling Step** (`sample_step`): Implement the reverse diffusion process to generate images.

## Catalogs

- `data/`: MNIST dataset (downloaded automatically)
- `diffusion_model.pt`: Saved model checkpoint

## What to Submit

You should submit a single .pdf file that contains the following:

1. A brief post-lab write-up that contains:
   
   a. A brief description of the diffusion model and its key components.
   
   b. Screenshots of your generated images and the training loss curve.
   
   c. A brief (couple of sentences) reflection on your take-aways from this lab exercise.
   
   d. Your completed Jupyter Notebook

## Important

As long as you implemented the model correctly, you do not need to train it to perfect convergence. Please do not stress if the generated images are not perfect. Train for as many epochs as you can. The longer you train, the better the results, but we **do not** evaluate your model's image quality, but your understanding of diffusion models.

## Key Concepts

### Forward Process (Adding Noise)
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

### Reverse Process (Denoising)
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

### Training Objective
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - The original DDPM paper by Ho et al.
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) - An excellent tutorial on diffusion models.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) - Hugging Face's tutorial on implementing diffusion models.
