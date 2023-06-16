import math

import torch.nn.functional as F
from torch import Tensor, nn

from src.utils import log_error


class SelfAttention(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
        bias: bool,
        dropout: float,
    ) -> None:
        """Do the same as multi-head attention but with a single matrix multiplication.

        Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int | None
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without remainder
        num_heads : int
            how many self-attention heads to use
        bias : bool
            whether to use bias or not: without bias might be a bit better and faster (but it's not for sure)
        dropout : float
            how many connection between tokens are dropped during each forward pass

        Raises
        ------
        ValueError
            if `embeddings_size` cannot be divided by `num_heads` without remainder
        """
        super().__init__()

        if not head_size:
            if embeddings_size % num_heads != 0:
                log_error(
                    "Embeddings size should be divisible by the number of heads without a residual, "
                    f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}",
                )
            head_size = embeddings_size // num_heads

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout

        # key, query and value projections (hence `3 * ...`) for all heads in a single batch
        self.qkv = nn.Linear(embeddings_size, 3 * self.head_size * self.num_heads, bias=self.bias)
        # output projection
        self.projection = nn.Linear(self.head_size * self.num_heads, embeddings_size, bias=self.bias)
        # regularization
        self.attention_dropout = nn.Dropout(self.dropout)
        self.projection_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Do multi-head attention in a single pass.

        Multiply by weight matrix -> split the result into query, key and value -> reshape each one of them
        into shape (batch, num_heads, time-steps, head_size). The rest is similar to single self-attention head
        forward pass.

        Parameters
        ----------
        x : Tensor
            input tensor of shape (batch, time-step, embedding size)

        Returns
        -------
        Tensor
            output tensor of the same shape as input: (batch, time-step, embedding size)
        """
        # notation:
        # - B  | batch
        # - T  | time-step (sequence length)
        # - C  | embeddings size
        # - hs | head size
        # - nh | number of heads

        B, T, C = x.shape  # noqa: N806

        # TODO: apply einops
        # single pass for query, key and value; that's why we need to split into 3 parts
        query, key, value = self.qkv(x).split(
            self.head_size * self.num_heads,
            dim=-1,
        )  # (B, T, C) -> (B, T, 3 * hs * nh) -> (B, T, hs * nh)

        # transform (B, T, nh * hs) -> (B, nh, T, hs) so it's similar to multi-head attention
        key = key.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        query = query.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        value = value.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # to obtain attention scores first do dot product of query and key
        attention_scores = query @ key.mT  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # In order to preserve 1 unit variance of the dot product of two vectors
        # we need to divide by square root of the features size (in our case - attention head size)
        # We need it to make sure that the values after softmax are well spread out, otherwise in worst
        # case scenario the values after the softmax will converge to one-hot encoding (like [0, 0, 1]) and
        # that will mean that the attention will be on a single (or couple of) tokens, and we want it to be
        # spread out (like [0.2, 0.1, 0.7])
        # we want to aggregate information not from a single node
        attention_scores /= math.sqrt(key.shape[-1])  # (B, nh, T, T)

        # since we want to do weighted averaging we need to transform attention scores into range [0, 1]
        # and sum of all scores should be equal to 1; softmax is a good tool for it
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, nh, T, T)

        # randomly prevent some nodes from communicating, some of theme randomly are set to zero
        # helps prevent overfitting
        attention_scores = self.attention_dropout(attention_scores)  # (B, nh, T, T)

        # perform the weighted aggregation of the values
        output = attention_scores @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        output = output.transpose(1, 2).reshape(B, T, self.head_size * self.num_heads)  # (B, T, hs * nh)
        # output projection
        output = self.projection(output)  # (B, T, C)

        return self.projection_dropout(output)  # (B, T, C)
