"""
We define a decoder-only transformer model for the task.

Pipeline for a decoder-only transformer:
    1.  Tokenization
        We will use the BERT tokenizer for now.
    2.  Add a CLS token
    3.  Add positional parameter to the encoding
        Classical would be to sample from a sine-wave.
    4.  Embedding
    5.  Blocks of:
            - Attention
            - MLP
            - Activation
    6.  Head
        Just a linear layer with the number of tokens as output dim.

For the Attention part:
    1.  Each token emits three vectors, Q, K and V.
        These vectors are derived using a simple linear layer (without a bias) each.
        So we need three projection functions: F_Q, F_K, F_V.
    2.  The emited vectors are combined into a matrix each, so M_Q, M_K and M_V.
    3.  We calculate M_Q x M_K^T in order to get the attention matrix
    4.  We apply at row-wise softmax to the attention matrix
    5.  We scale the attention matrix
    6.  We calculate the product of attention and value matrix

Dimensions:
    - d_model: The length of the embedding vector for each token.
"""
