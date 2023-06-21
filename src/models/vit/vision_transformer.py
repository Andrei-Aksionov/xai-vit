import re

import torch
from loguru import logger
from torch import Tensor, nn

from src.models.vit.embeddings import PatchEmbeddings
from src.models.vit.transformer_block import LayerNorm, TransformerBlock
from src.utils.error import log_error


class ViT(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        head_size: int | None,
        num_heads: int,
        feed_forward_scaling: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        num_classes: int,  # TODO: docstring argument
        num_channels: int,
        patch_size: int,
        image_size: int,
    ) -> None:
        # TODO: update docstring
        """Create Generative Pre-trained Transformer model (decoder part of transformer architecture).

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
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

        self.embeddings_size = embeddings_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.feed_forward_scaling = feed_forward_scaling
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size

        # Create patch embeddings
        patch_embeddings = PatchEmbeddings(
            patch_size=self.patch_size,
            in_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
        )
        self.context_size = patch_embeddings.get_num_patches(self.image_size)

        self.embeddings = nn.ParameterDict(
            {
                "patch_embeddings": patch_embeddings,
                "cls_token": nn.Parameter(torch.zeros(1, 1, self.embeddings_size)),
                "position_embeddings": nn.Parameter(torch.randn(1, self.context_size + 1, self.embeddings_size) * 0.02),
            },
        )

        self.embeddings_dropout = nn.Dropout(self.dropout) if self.dropout else nn.Identity()

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embeddings_size=self.embeddings_size,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    bias=self.bias,
                    dropout=self.dropout,
                    feed_forward_scaling=self.feed_forward_scaling,
                )
                for _ in range(self.num_layers)
            ],
        )

        self.layernorm = LayerNorm(self.embeddings_size, bias=self.bias)  # final layer norm

        self.classifier = nn.Linear(self.embeddings_size, self.num_classes)

        # report number of parameters
        logger.debug(
            "ViT model is created with number of parameters: {:.2f} million".format(
                sum(param.numel() for param in self.parameters()) / 1e6,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # TODO: update docstring
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
        x = self.embeddings.patch_embeddings(x)
        B, T, C = x.shape

        # add CLS token and positional embeddings
        cls_token = self.embeddings.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.embeddings.position_embeddings

        # apply transformer
        x = self.embeddings_dropout(x)
        x = self.transformer_blocks(x)

        # classification step
        cls = x[:, 0, :]
        return self.classifier(cls)

    @classmethod
    def from_pretrained(cls: "ViT", vit_type: str) -> "ViT":
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
        from transformers import ViTConfig, ViTForImageClassification

        # check that the gpt2 type is supported
        supported_types = "google/vit-base-patch16-224"
        if vit_type not in supported_types:
            logger.warning(f"Only '{supported_types}' were tested, but '{vit_type}' was provide.")

        # prepare config that will be passed into our GPT implementation
        source_config = ViTConfig.from_pretrained(vit_type)
        # syncing argument names between our GPT implementation and from Huggingface
        # TODO: fix ordering
        target_config = {
            "embeddings_size": source_config.hidden_size,
            "head_size": None,
            "num_heads": source_config.num_attention_heads,
            "feed_forward_scaling": source_config.intermediate_size // source_config.hidden_size,
            "num_layers": source_config.num_hidden_layers,
            # TODO: in config it's a qkv bias
            "bias": source_config.qkv_bias,
            # TODO: there are at least two different dropouts
            "dropout": source_config.hidden_dropout_prob,
            "num_classes": 1_000,
            "num_channels": source_config.num_channels,
            "patch_size": source_config.patch_size,
            "image_size": source_config.image_size,
        }

        # Instantiate GPT model and extract params
        logger.debug("Creating ViT model with parameters: {}".format(target_config))
        target_model = ViT(**target_config)
        # extract gpt model parameters into a variable
        target_state_dict = target_model.state_dict()

        # TODO: source model should be placed before target one
        # create Huggingface pretrained GPT2 model
        logger.debug("Loading pretrained Huggingface model of size '{}' ...".format(vit_type))
        source_model = ViTForImageClassification.from_pretrained(vit_type)
        logger.debug("Huggingface model is loaded.")
        source_state_dict = source_model.state_dict()

        # since names of layers are different for our implementation and the one from Huggingface,
        # we need to map them properly
        param_mapping = (
            (r"vit\.", ""),
            (r"encoder\.layer", "transformer_blocks"),
            (r"attention\.attention", "attention"),
            (r"attention\.output\.dense", "attention.output"),
            (r"intermediate\.dense", "feed_forward.intermediate"),
            (r"(\d)(\.output\.dense)", r"\g<1>.feed_forward.output"),
        )

        def sync_name(name: str) -> str:
            for pattern, replacement in param_mapping:
                name = re.sub(pattern, replacement, name)
            return name

        # loading weights: step 1 of 2
        for source_key in source_state_dict:
            # in Huggingface implementation query, key and value matrices are
            # stored separately, while in this implementation - in a combined qkv matrix.
            # So for now skip corresponding weights - it will be done later
            if any(key in source_key for key in ("query", "key", "value")):
                continue
            # map param name from Hugginface notation to this implementation's notation
            target_key = sync_name(source_key)
            source_weights = source_state_dict[source_key]
            if source_weights.shape != target_state_dict[target_key].shape:
                log_error(
                    f"Shape mismatch: for '{target_key}' shape of source '{source_weights.shape}' and destination - "
                    f"'{target_state_dict[target_key].shape}'",
                )
            with torch.no_grad():
                target_state_dict[target_key].copy_(source_weights)

        # loading weights: step 2 of 2
        # combine weights for query, key and value along embeddings dimension and copy
        # into ours qkv matrix
        for idx in range(target_model.num_layers):
            source_key_template = f"vit.encoder.layer.{idx}.attention.attention.{{matrix}}.{{weight}}"
            target_key_template = f"transformer_blocks.{idx}.attention.qkv.{{weight}}"

            for weight_type in ("weight", "bias"):
                source_weights = torch.concat(
                    [
                        source_state_dict[source_key_template.format(matrix=m, weight=weight_type)]
                        for m in ("query", "key", "value")
                    ],
                    dim=0,
                )
                target_key = target_key_template.format(weight=weight_type)

                with torch.no_grad():
                    target_state_dict[target_key].copy_(source_weights)

        logger.debug("Weights are copied.")

        return target_model
