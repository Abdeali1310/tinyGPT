# TinyGPT

A minimal GPT-style decoder-only transformer implementation built from scratch in PyTorch.

This project reimplements the core architecture introduced in the paper *Attention Is All You Need* and follows the decoder-only transformer approach used in GPT-style language models.

Reference Paper:
https://arxiv.org/abs/1706.03762

## Overview

The project implements a character-level autoregressive language model trained on the Tiny Shakespeare dataset (input.txt).

The model includes:

* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head attention
* Feedforward networks
* Residual connections
* Layer normalization
* Dropout regularization
* Transformer blocks
* Autoregressive text generation

The architecture is implemented entirely in PyTorch without using pretrained models or external transformer libraries.

## Architecture

The model is based on a decoder-only transformer.

Unlike encoder-decoder transformers originally proposed for machine translation, this implementation uses masked self-attention to ensure tokens can only attend to previous tokens during generation.

The training objective is next-token prediction using autoregressive language modeling.

## Training Details

Dataset: Tiny Shakespeare
Tokenizer: Character-level tokenizer
Model Parameters: ~10 Million Parameters

Training Metrics:

* Final Train Loss: ~1.11
* Final Validation Loss: ~1.47

Training was performed on Google Colab using GPU acceleration.
The final transformer configuration required approximately 40–45 minutes to train.


## Train on Your Own Dataset

1. Replace the contents of `input.txt` with your own text dataset.

2. Run the GPT training script:

```bash
python gpt.py
```

3. The model will automatically:

* build a character-level vocabulary
* tokenize the dataset
* train the transformer model
* generate text samples

You can experiment with the following hyperparameters inside `gpt.py`:

* `block_size`
* `n_embd`
* `n_layer`
* `n_heads`
* `dropout`

to create larger or smaller transformer configurations.


## Pretrained model 
A pretrained model is trained on the Tiny Shakespeare dataset is included in the repository(tinygpt_model.pth), so you can directly try text generation without training from scratch by running 

```bash
python inference.py
```

Training the model again will overwrite the existing checkpoint file with the newly trained weights unless you save it using a different filename.



## Text Generation / Inference

Once the model is trained, the saved weights can be loaded again for text generation without retraining the entire model.

You can give the model an initial prompt, and it will continue generating text token-by-token based on the learned patterns from the training dataset.

Run inference using:

```bash
python inference.py
```


All in all this project trains a GPT-style transformer model from scratch to learn patterns from given text data and generate new text in a similar style to the training dataset.
