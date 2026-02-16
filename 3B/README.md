# Transformer
This demo showcases training a **Transformer** based **Large Language Model (LLM)**. 

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), it is designed to be simple and easy to understand, making it an excellent starting point for beginners learning how to train an LLM from scratch using PyTorch.

The demo is trained on a 1961 kB [Shakespear work](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, and the model size is about 51,430kB. I trained this model on a single NVIDIA GeForce RTX 3050 Ti GPU in around 30 minutes and the result is approximately 10,690,625 parameters.


## Install and start
1. Install dependencies.

To install dependencies, you can either update your local environment with the script below:
```bash
conda env update --file environment.yml --prune
```
Or install new environment:
```bash
conda env create -f environment.yml
```
The `create` script will create a new environment `engs106_1`.

If you prefer, you can install all the dependencies manually. These are the dependancies I used:
- cuda-version=12.4
- cuda=12.4.1
- pytorch
- pytorch-cuda=12.4
- torchaudio
- torchvision
- tqdm

2. Modify the script in `lab3B.ipynb`.

The model will start training on the dataset. Training & validation `losses` will be printed on the console screen using `tqdm`.

## Catalogs

- `data/*`: Houses the sample dataset used for training
- `model/*`: The model saving path
- `utils.py`: Transformer model logic code

### What to Submit
You should submit a single .pdf file that contains the following:
1. A brief post-lab write-up that contains the following for each part of this assignment:
    a. A brief description of your model. Justify your selection of model parameters.
    
    b. An evaluation of your model, including evidence as appropriate.
    
    c. A brief (couple of sentences) reflection on your take-aways from this lab exercise.

    d. Jupyter Notebook

### Important
As long as you implemented the model correctly, you do not need to train it to convergence. Please do not stress if the model does not perform well. Train as big and as long as you can. The bigger and the longer, the better, but we **do not** evaluate your model's performance, but your understanding.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) The original paper of Transformer architecture.
+ [nanoGPT](https://github.com/karpathy/nanoGPT) Andrej Karpathy's famous video tutorial on how to build a GPT model from scratch.
* [Transformers from Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html) A clear and easy implementation of Andrej's video contents by Mat Miller.
