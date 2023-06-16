from torch import Tensor, nn

from src.models.vit.attention import SelfAttention


# TODO: perhaps rename FeedForward to MLP?
class FeedForward(nn.Module):
    def __init__(self, embeddings_size: int, bias: bool, scaling: int, dropout: float) -> None:
        """Apply on per-token level. Each token is processed independently.

        If the is no feed-forward layer, self-attention is simply a process of re-averaging of value vectors. In order
        to add element-wise non-linearity transformation of incoming vectors we add feed-forward part.

        You can think about it in this way:
        - attention step is for communication between tokens
        - feed-forward is for processing this information (of how tokens are related to each other via attention)

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `scaling` specifies by how much
        dropout : float
            how many connection between tokens are dropped during each forward pass
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.bias = bias
        self.scaling = scaling
        self.dropout = dropout

        self.c_fc = nn.Linear(self.embeddings_size, self.scaling * self.embeddings_size, bias=self.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.projection = nn.Linear(self.scaling * self.embeddings_size, self.embeddings_size, bias=self.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # TODO: redo it as a nn.Sequential
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.projection(x)
        # TODO: check that we need that last dropout
        return self.dropout(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: tuple[int] | list[int], bias: bool, **kwargs) -> None:
        """Wrap `torch.nn.LayerNorm` to have ability to disable bias.

        Parameters
        ----------
        normalized_shape : tuple[int] | list[int]
            number of features of the layer
        bias : bool
            whether to use bias or not
        **kwargs : dict
            keyword arguments that are expected by `torch.nn.LayerNorm`: [eps, elementwise_affine, device, dtype]
        """
        super().__init__(normalized_shape, **kwargs)
        if not bias:
            self.bias = None


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
        bias: bool,
        dropout: float,
        feed_forward_scaling: int,
    ) -> None:
        """Create transformer block with self-attention, layer normalization and feed-forward.

        Self-attention is used in order to add communication between tokens, feed-forward - for
        processing this information. Layer normalization allows to build deeper neural networks.
        `Note`: pre-normalization of layers is used here.

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
            without residual
        num_heads : int
            how many self-attention heads to use
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster (but it's not for sure)
        dropout : float
            how many connection between tokens are dropped during each forward pass
        feed_forward_scaling: int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout
        self.feed_forward_scaling = feed_forward_scaling

        attention_kwargs = {
            "embeddings_size": self.embeddings_size,
            "context_size": self.context_size,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "bias": self.bias,
            "dropout": self.dropout,
        }

        self.self_attention = SelfAttention(**attention_kwargs)

        self.feed_forward = FeedForward(
            embeddings_size=self.embeddings_size,
            bias=self.bias,
            scaling=self.feed_forward_scaling,
            dropout=self.dropout,
        )
        self.layer_norm_1 = LayerNorm(normalized_shape=self.embeddings_size, bias=self.bias)
        self.layer_norm_2 = LayerNorm(normalized_shape=self.embeddings_size, bias=self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer block with layer norm, self-attention and feed-forward.

        `+` sign is for residual connection (allows to build deeper neural nets)

        Parameters
        ----------
        x : Tensor
            input tensor of size (batch_size, time-steps, channels_num)

        Returns
        -------
        Tensor
            output tensor of size (batch_size, time-steps, channels_num)
            output has the same size as input
        """
        # + sign is used for residual connection
        # helps with gradient flow and allows to build deeper neural nets
        # TODO: in huggingface repo they use LayerScale. Do I need one?
        x = x + self.self_attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))
