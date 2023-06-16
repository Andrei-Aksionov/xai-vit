import math
from functools import reduce

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from src.models.vit.transformer_block import LayerNorm, TransformerBlock
from src.utils.error import log_error


class ViT(nn.Module):
    def __init__(
        self,
        # TODO: create a config class
        # vocab_size: int,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
        feed_forward_scaling: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        num_classes: int,  # TODO: docstring argument
        num_channels: int,
        patch_size: int,
        # weight_tying: bool = True,
        # weight_decay: float | None = None,
    ) -> None:
        """Create Generative Pre-trained Transformer model (decoder part of transformer architecture).

        Parameters
        ----------
        vocab_size : int
            number of unique tokens in vocabulary
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int | None
            the size of output of self-attention
        num_heads : int
            how many self-attention heads to use
        feed_forward_scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        num_layers : int
            how many transformer blocks to use
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        dropout : float
            how many connection between tokens are dropped during each forward pass
        weight_tying: bool
           Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and
           softmax layers. This method also massively reduces the total number of parameters in the language models that
           it is applied to.
           https://paperswithcode.com/method/weight-tying, by default True
        weight_decay: float | None
            if provided will prepare parameters for optimizer
        """
        super().__init__()

        # self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        # TODO: context size is calculated during patch conversion
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.feed_forward_scaling = feed_forward_scaling
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.num_classes = num_classes
        # self.weigh_tying = weight_tying
        # self.weight_decay = weight_decay
        # TODO: this one needs to be calculated
        self.num_channels = num_channels
        self.patch_size = patch_size

        # TODO: input layer that transforms input embedding size to model's embeddings size
        self.input_layer = nn.Linear(self.num_channels * self.patch_size**2, self.embeddings_size)

        # TODO: change it to patch embeddings
        # self.token_embedding_table = nn.Embedding(self.vocab_size, self.embeddings_size)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (context_size) tokens

        # TODO: change positional embeddings
        # TODO: can we use here nn.Embeddings?
        # NOTE: self.context_size + 1 because we need account cls token
        self.positional_embedding_table = nn.Parameter(
            torch.randn(1, self.context_size + 1, self.embeddings_size) * 0.02,
        )
        # self.positional_embedding_table = nn.Embedding(self.context_size, self.embeddings_size)
        # TODO: add cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embeddings_size))
        self.embeddings_dropout = nn.Dropout(self.dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embeddings_size=self.embeddings_size,
                    context_size=self.context_size,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    bias=self.bias,
                    dropout=self.dropout,
                    feed_forward_scaling=self.feed_forward_scaling,
                )
                for _ in range(self.num_layers)
            ],
        )
        self.layer_norm_final = LayerNorm(self.embeddings_size, bias=self.bias)  # final layer norm

        # TODO: change it to classification head
        # self.language_model_head = nn.Linear(self.embeddings_size, self.vocab_size, bias=False)
        self.head = nn.Linear(self.embeddings_size, self.num_classes)

        # TODO: is it even a thing for ViT
        # if self.weigh_tying:
        #     self.token_embedding_table.weight = self.language_model_head.weight

        # self.apply(self.__init_weights)
        # TODO: revisit it (it's from GPT-2 paper)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for name, param in self.named_parameters():
        #     if name.endswith("projection.weight"):
        #         torch.nn.init.normal_(param, mean=0.0, std=0.2 / math.sqrt(2 * self.num_layers))

        # configure parameters for optimizer that will be decay and that will not
        # if self.weight_decay:
        #     self.optimizer_parameters = self.__optimizer_parameters(weight_decay=self.weight_decay)

        # report number of parameters
        logger.debug(
            "GPT language model is created with number of parameters: {:.2f} million".format(
                self.__get_parameters_number(False) / 1e6,
            ),
        )

    def __get_parameters_number(self, exclude_positional_embeddings: bool = True) -> int:
        # TODO: print total number of parameters and number of learnable parameters
        """Return total number of parameters of the model without counting parameters of positional embeddings."""
        params_count = sum(param.numel() for param in self.parameters())
        if exclude_positional_embeddings:
            # FIXME: this doesn't work, says "Parameter has no attribute weight"
            params_count -= self.positional_embedding_table.weight.numel()
        return params_count

    def __init_weights(self, module: torch.nn.modules) -> None:
        """Initialize Embedding and Linear layers with a smaller std.

        By default weights of Embedding layer are initialized from normal distribution
        with zero mean and unit std ( N(0, 1) ).
        Weights for linear layer are initialized from uniform distribution from
        [-k**0.5, k**0.5], where k = 1 / in_features. Even with 128 features it will be
        [-0.09, 0.09].

        Parameters
        ----------
        module : torch.nn.modules
            module of the network
        """
        if isinstance(module, nn.Embedding | nn.Linear):
            # TODO: check different std init works better
            #   0.2 / sqrt ( 2 * number of transformer blocks)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Do the whole forward pass for decoder part of transformer.

        This forward method includes all steps for decoder:
        1. token embeddings + positional
        2. transformer block consisting of self-attention, feed-forward, addNorm
        3. logits for each token in vocabulary (or the last one in case of inference)

        Parameters
        ----------
        idx : Tensor
            tensor of size (batch, time-step) consisting of indices of tokens inside vocabulary
            for each time-step for each batch
        inference: bool
            during inference we don't care about all tokens but the very last one, so we can
            apply final language head only on the last token and save some computations

        Raises
        ------
        ValueError
            if there is a mismatch between number of time-steps and self.context_size

        Returns
        -------
        Tensor
            tensor of size (batch, time-step, vocabulary_size): logits for each token in vocabulary
            for each time-step for each batch, or the last one in case of inference
        """
        # batch, time-step
        # B, T = idx.shape  # noqa: N806
        # if self.context_size < T:
        #     msg = f"Cannot do forward pass on sequence of length {T}, "
        #     f"context size should less or equal to {self.context_size}"
        #     raise ValueError(msg)

        # # obtain token embeddings and add positional information
        # token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        # # if kv_cache is provided and it's not an empty tensor
        # positional_embeddings = self.positional_embedding_table.weight[:T]  # (T, C)
        # x = token_embeddings + positional_embeddings  # (B, T, C)
        # x = self.embeddings_dropout(x)  # (B, T, C)

        # # apply multiple transformer blocks
        # for block in self.transformer_blocks:
        #     x = block(x)
        # # apply final normalization and generate logits for each token in vocabulary
        # x = self.layer_norm_final(x)  # (B, T, C)

        # # during inference we don't need to encode all token predictions,
        # # only the last one (newly generated)
        # if inference:
        #     return self.language_model_head(x[:, -1:, :])  # (B, 1, vocab_size)
        # return self.language_model_head(x)  # (B, T, vocab_size)

        B, T, C = x.shape
        # B, T, C_1, C_2 = x.shape
        print(f"1: {x.shape=}")

        # from input channels to embeddings_size
        x = self.input_layer(x)
        print(f"2: {x.shape=}")

        # add CLS token and positional embeddings
        cls_token = self.cls_token.repeat(B, 1, 1)
        print(f"3: {cls_token.shape=}")
        x = torch.cat([cls_token, x], dim=1)
        print(f"4: {x.shape=}")
        # TODO: why T + 1?
        x = x + self.positional_embedding_table[:, : T + 1]
        print(f"5: {x.shape=}")

        # apply transformer
        x = self.embeddings_dropout(x)
        print(f"6: {x.shape=}")
        # TODO: do we need it here?
        x = x.transpose(0, 1)
        # TODO: change it to sequential
        for block in self.transformer_blocks:
            x = block(x)
        print(f"7: {x.shape=}")

        # classification step
        cls = x[0]
        print(f"8: {cls.shape=}")
        predictions = self.head(cls)
        print(f"9: {predictions.shape=}")
        return predictions

    def __optimizer_parameters(self, weight_decay: float) -> tuple[dict, dict]:
        """Configure optimizer with weight decay for nn.Linear.

        Parameters
        ----------
        weight_decay : float
            weight decay (L2 penalty)

        Returns
        -------
        tuple[dict, dict]
            list of two dictionaries, containing parameter names that will have weight decay
            and that will not accordingly

        Raises
        ------
        ValueError
            if model's layer are not in (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding)
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay, no_decay = set(), set()
        expected_weight_modules = (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding)
        for pn, _ in self.named_parameters():
            # get the parent module by the parameter's name
            module = reduce(lambda module, key: getattr(module, key), pn.split(".")[:-1], self)
            if type(module) not in expected_weight_modules:
                log_error(f"Expected the module to be one of '{expected_weight_modules}', but got {type(module)}")
            if isinstance(module, nn.Linear) and pn.endswith("weight"):
                decay.add(pn)
            else:
                no_decay.add(pn)

        # create the pytorch optimizer object
        param_dict = dict(self.named_parameters())
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Prepare tensors to be compatible with Pytorch's Cross-Entropy loss and applies it.

        Cross-Entropy expects to have features as the second-dimension, so we first need to
        transform logits in supported shape and in order to align targets tensor with logits tensor,
        we need to transform targets too.

        Parameters
        ----------
        logits : Tensor
            tensor with model's outputs
        targets : Tensor
            tensor with true labels

        Returns
        -------
        Tensor
            tensor with loss value (of how good model's predictions are)
        """
        B, T, C = logits.shape  # noqa: N806
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
        )

    @classmethod
    def from_pretrained(cls: "GPTLanguageModel", gpt2_type: str) -> "GPTLanguageModel":
        """Create GPT2 model with weights copied from Huggingface pretrained model.

        Parameters
        ----------
        gpt2_type : str
            GPT2 type: gpt2, gpt2-medium, gpt2-large and gpt2-xl are supported

        Returns
        -------
        GPTLanguageModel
            a model with pretrained weights

        Raises
        ------
        ValueError
            provided gpt2 type is not in the list of supported types
        ValueError
            Huggingface GPT2 config has different values for dropout
        ValueError
            mismatch number of keys/parameters between GPT and Huggingface's GPT2
        ValueError
            mismatch shape of a parameter between GPT and Huggingface's GPT2
        """
        # Notation:
        # target* | the model to which the weights are copied (this GPT implementation)
        # source* | the model from which the weight are copied (Huggingface GPT2 implementation)

        # huggingface transformers library is needed only in this method
        from transformers import GPT2Config, GPT2LMHeadModel

        # check that the gpt2 type is supported
        supported_types = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        if gpt2_type not in supported_types:
            log_error(f"Only '{supported_types}' are supported, but '{gpt2_type}' was provided.")

        # prepare config that will be passed into our GPT implementation
        source_config = GPT2Config.get_config_dict(gpt2_type)[0]
        # syncing argument names between our GPT implementation and from Huggingface
        target_config = {
            "vocab_size": source_config["vocab_size"],
            "embeddings_size": source_config["n_embd"],
            "context_size": source_config["n_ctx"],
            "num_layers": source_config["n_layer"],
            "num_heads": source_config["n_head"],
            "head_size": None,
            "feed_forward_scaling": 4,
            "bias": True,
        }
        # Dropouts for embedding, attention, residual, and summary in Huggingface implementation have to be identical
        dropouts = [(name, value) for name, value in source_config.items() if "dropout" in name]
        if any(dropouts[0][1] != x[1] for x in dropouts[1:]):
            log_error(f"All dropouts for GPT2 model should have had the same value, but in fact received '{dropouts}'")
        target_config["dropout"] = dropouts[0][1]

        # Instantiate GPT model and extract params
        logger.debug("Creating GPT model with parameters: {}".format(target_config))
        target_model = GPTLanguageModel(**target_config)
        # extract gpt model parameters into a variable
        target_state_dict = target_model.state_dict()
        # drop tril as it's a buffer (doesn't learn anything)
        target_state_dict_keys = [k for k in target_state_dict if not k.endswith(".self_attention.tril")]

        # create Huggingface pretrained GPT2 model
        logger.debug("Loading pretrained Huggingface model of size '{}' ...".format(gpt2_type))
        source_model = GPT2LMHeadModel.from_pretrained(gpt2_type)
        logger.debug("Huggingface model is loaded.")
        source_state_dict = source_model.state_dict()
        # skip bias as it's not a parameter
        source_state_dict_keys = [
            key for key in source_state_dict if not key.endswith((".attn.bias", ".attn.masked_bias"))
        ]

        if len(target_state_dict_keys) != len(source_state_dict_keys):
            log_error(f"Mismatch number of keys: {len(target_state_dict_keys)} != {len(source_state_dict_keys)}")

        # since names of layers are different for our implementation and the one from Huggingface,
        # we need to map them properly
        param_mapping = {
            "transformer": "",
            "wte": "token_embedding_table",
            "wpe": "positional_embedding_table",
            "h": "transformer_blocks",
            "attn": "self_attention",
            "c_attn": "causal_self_attention",
            "ln_1": "layer_norm_1",
            "ln_2": "layer_norm_2",
            "c_proj": "projection",
            "mlp": "feed_forward",
            "ln_f": "layer_norm_final",
            "lm_head": "language_model_head",
        }

        def sync_name(name: str) -> str:
            names_renamed = [param_mapping.get(n, n) for n in name.split(".")]
            return ".".join(filter(None, names_renamed))

        # in Huggingface implementation attention and feed forward use 'Conv1D',
        # while in this implementation - LinearLayer
        # that means that we can use weights for those layers, only we need to transpose them before copying
        to_transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

        # loading weights
        logger.debug("Starting copying weights from pretrained Huggingface model into our implementation ...")
        for source_key in source_state_dict_keys:
            # map param name from Hugginface notation to this implementation's notation
            target_key = sync_name(source_key)
            source_weights = source_state_dict[source_key]
            if source_key.endswith(to_transposed):
                source_weights = source_weights.t()
            if source_weights.shape != target_state_dict[target_key].shape:
                log_error(
                    f"Shape mismatch: shape of source '{source_weights.shape}' and destination - "
                    f"'{target_state_dict[target_key].shape}'",
                )
            with torch.no_grad():
                target_state_dict[target_key].copy_(source_weights)
        logger.debug("Weights are copied.")

        return target_model
