"""
We define a decoder-only transformer model for the task.

Pipeline for a decoder-only transformer:
    1.  Tokenization
        We will use the BERT tokenizer for now.
    2.  Add a CLS token
    3.  Add positional parameter to the encoding
        Classical would be to sample from a sine-wave,
        but we are going to use a learned positional embeddings instead since the only
        potential disadvantage of such an approach listed in the "Attention is all you need"
        paper does not apply to sequences as small as ours.
    4.  Embedding
    5.  Blocks of:
            - Attention
            - MLP
            - Activation
    6.  Head
        We map the last hidden state of the cls token to the number of classes (m,f,n)

For the Attention part:
    1.  Each token emits three vectors, Q, K, and V.
        These vectors are derived using a simple linear layer (without a bias) each.
        So we need three projection functions: F_Q, F_K, F_V.
    2.  The emitted vectors are combined into a matrix each, so M_Q, M_K, and M_V.
    3.  We calculate M_Q x M_K^T in order to get the attention matrix.
    4.  We apply a row-wise softmax to the attention matrix.
    5.  We scale the attention matrix.
    6.  We calculate the product of the attention matrix and the value matrix.

Dimensions:
    - d_model: The length of the embedding vector for each token.
"""

import torch
from torch import nn
from src.tokenizer import tokenizer, tokenize_texts
import math


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # A learnable parameter for the positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(max_sequence_length, d_model))
        # Initialize the positional encoding with a small random value
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # the embeddings for each token
        self.embeddings = nn.Embedding(tokenizer.vocab_size, d_model)

        # initialize the transformers layers
        self.layers = nn.ModuleList(
            [AttentionBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        # and finally a classification head
        self.head = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout()

    def forward(self, x: list[str]) -> torch.Tensor:
        # tokenize the texts (also appends the cls token)
        x = tokenize_texts(x)

        # move to the same device as the model
        x = x.to(next(self.parameters()).device)

        # apply embedding
        x = self.embeddings(x)

        # add positional encodings
        x = x + self.pos_encoding[: x.shape[1]]

        # dropout
        x = self.dropout(x)

        # apply attention blocks
        for attn_block in self.layers:
            x = attn_block(x)

        # extract the cls tokens
        cls_tokens = x[:, -1, :]

        # and apply the classification head
        logits = self.head(cls_tokens)

        # and a softmax to get probabilities
        return logits.softmax(dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        # create the attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_k) for _ in range(num_heads)]
        )

        # linear layer after attention
        self.attn_lin = nn.Linear(d_model, d_model)

        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )

        # Dropout layers
        self.dropout = nn.Dropout()  # default dropout prop is 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply each attention head and concatenate the output
        attn = torch.concat([head(x) for head in self.heads], dim=2)
        attn = self.attn_lin(attn)
        attn = self.dropout(attn)

        # norm and residual
        x = self.norm1(attn + x)

        # MLP
        mlp_out = self.mlp(x)
        mlp_out = self.dropout(mlp_out)

        # norm and residual again
        x = self.norm2(mlp_out + x)

        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k

        # We define three linear function (Q, K, V)
        self.Q = nn.Linear(d_model, d_k, bias=False)
        self.K = nn.Linear(d_model, d_k, bias=False)
        self.V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape: (batch_size, seq_length, d_model)

        # calculate queries, keys, and values
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        # calculate the attention matrix
        # (batch_size, seq_length, seq_length)
        attn = queries @ keys.transpose(-2, -1)

        # scale the attention matrix
        attn = attn / math.sqrt(self.d_k)

        # apply row-wise softmax (make sure to specify the correct row dimension)
        attn = attn.softmax(dim=-1)

        # multiply attn with the value matrix to get the output
        # (batch_size, seq_length, d_k)
        return attn @ values
