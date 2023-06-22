import re
from collections import defaultdict

import torch
from loguru import logger
from torch import Tensor, nn

from src.models.vit.embeddings import PatchEmbeddings
from src.models.vit.transformer_block import LayerNorm, TransformerBlock
from src.utils.error import log_error


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        patch_size: int,
        embeddings_size: int,
        head_size: int | None,
        num_heads: int,
        feed_forward_scaling: int,
        bias: bool,
        dropout: float,
        num_layers: int,
        num_classes: int,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Create Vision Transformer model.

        The architecture mimics what is used in Google's ViT model with patch size of 16 and image size of 224.
        https://huggingface.co/google/vit-base-patch16-224

        Parameters
        ----------
        image_size: int
            size of the image that will be used for the model during training/inference
        num_channels: int
            number of channels of the input tensor
        patch_size: int
            the input tensor will be splitted into patches of this size
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        head_size : int | None
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without remainder
        num_heads : int
            how many self-attention heads to use
        feed_forward_scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        bias: bool
            whether to use bias or not
        dropout : float
            how many connection between tokens are dropped during each forward pass
        num_layers : int
            how many transformer blocks to use
        num_classes: int
            for how many classes the model should output predictions
        id2label: dict[int, str] | None
            if provided will be used for decoding predictions
        """
        super().__init__()

        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embeddings_size = embeddings_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.feed_forward_scaling = feed_forward_scaling
        self.bias = bias
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.id2label = id2label

        ### Embeddings ###
        # Create patch embeddings
        patch_embeddings = PatchEmbeddings(
            patch_size=self.patch_size,
            in_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
        )
        # +1 - to reserve space for classification token
        num_patches = patch_embeddings.get_num_patches(self.image_size) + 1
        self.embeddings = nn.ParameterDict(
            {
                "patch_embeddings": patch_embeddings,
                "cls_token": nn.Parameter(torch.zeros(1, 1, self.embeddings_size)),
                "position_embeddings": nn.Parameter(torch.randn(1, num_patches, self.embeddings_size)),
                "dropout": nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            },
        )

        ### Transformer blocks/layers ###
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

        # final layer norm - normalize data after the last transformer block
        self.layernorm = LayerNorm(self.embeddings_size, bias=self.bias)
        # the last layer - transform tokens into logits for each class
        self.classifier = nn.Linear(self.embeddings_size, self.num_classes)

        # report number of parameters
        logger.debug(
            "ViT model is created with number of parameters: {:.2f} million".format(
                sum(param.numel() for param in self.parameters()) / 1e6,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Do the whole forward pass for ViT - encoder part of transformer for image processing.

        Parameters
        ----------
        x: Tensor
            input image in form of tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        Tensor
            tensor of size (batch, num_classes) that contains logits for all classes
        """
        # Notation:
        # B - batch size
        # T - sequence length (number of patches)
        # C - embeddings size (of each patch)
        # in_C, H, W - number of channels of the image/tensor, height and width

        # transform tensor of shape (B, in_C, H, W) into a set of patches where each one
        # is represented by a vector of size `C`
        x = self.embeddings.patch_embeddings(x)  # (B, T, C)

        # add classification token (CLS) that will contain information for classification
        cls_token = self.embeddings.cls_token.repeat(x.size(0), 1, 1)  # (B, 1, C)
        x = torch.cat([cls_token, x], dim=1)  # (B, T + 1, C)
        # add information of position of each token
        x = x + self.embeddings.position_embeddings  # (B, T + 1, C)
        x = self.embeddings.dropout(x)  # (B, T + 1, C)

        # apply transformer blocks
        x = self.transformer_blocks(x)  # (B, T + 1, C)

        # classification step: extract CLS token and process it's information via classification layer
        cls = x[:, 0, :]  # (B, C)
        return self.classifier(cls)  # (B, num_classes)

    def decode_logits(self, logits: Tensor, top_k: int) -> dict[int, list[dict]]:
        """Return top k classes and corresponding probabilities.

        Parameters
        ----------
        logits : Tensor
            logits as model's output
        top_k : int
            classes with top k probabilities will be returned

        Returns
        -------
        dict[int, list[dict]]
            for each image list of top k classes and their probabilities
        """
        response = defaultdict(list)

        # convert logits into probabilities
        preds = torch.nn.functional.softmax(logits, dim=-1)
        # work only with top k probabilities
        values, indices = torch.topk(preds, k=top_k)
        for batch in range(preds.size(0)):
            for value, idx in zip(values[batch], indices[batch]):
                response[batch].append({"class": self.id2label[idx.item()], "probability": value.item()})

        return response

    @classmethod
    @torch.no_grad()
    def from_pretrained(cls: "ViT", vit_type: str) -> "ViT":
        """Create ViT model with weights copied from Huggingface pretrained model.

        Parameters
        ----------
        vit_type : str
            ViT type: google/vit-base-patch16-224 is supported for now, though other should work too,
            yet not tested

        Returns
        -------
        ViT
            a model with pretrained weights

        Raises
        ------
        ValueError
            mismatch shape of a parameter between ViT and Huggingface's ViT
        """
        # Notation:
        # target* | the model to which the weights are copied (this ViT implementation)
        # source* | the model from which the weight are copied (Huggingface ViT implementation)

        # huggingface transformers library is needed only in this method
        from transformers import ViTForImageClassification

        # check that the ViT type is supported
        supported_types = ("google/vit-base-patch16-224",)
        if vit_type not in supported_types:
            logger.warning(f"Only '{supported_types}' were tested, but '{vit_type}' was provide.")

        # create Huggingface pretrained ViT model
        logger.debug("Loading pretrained Huggingface model of size '{}' ...".format(vit_type))
        source_model = ViTForImageClassification.from_pretrained(vit_type)
        logger.debug("Huggingface model is loaded.")
        source_state_dict = source_model.state_dict()

        # prepare config that will be passed into our ViT implementation
        # syncing argument names between our ViT implementation and from Huggingface
        target_config = {
            "image_size": source_model.config.image_size,
            "num_channels": source_model.config.num_channels,
            "patch_size": source_model.config.patch_size,
            "embeddings_size": source_model.config.hidden_size,
            "head_size": None,
            "num_heads": source_model.config.num_attention_heads,
            "feed_forward_scaling": source_model.config.intermediate_size // source_model.config.hidden_size,
            "bias": source_model.config.qkv_bias,
            "dropout": source_model.config.hidden_dropout_prob,
            "num_layers": source_model.config.num_hidden_layers,
            "num_classes": source_model.classifier.out_features,
        }

        # Instantiate ViT model and extract params
        logger.debug("Creating ViT model with parameters: {}".format(target_config))
        # id2label is added here to not add noise in the logging
        target_config["id2label"] = source_model.config.id2label
        target_model = ViT(**target_config)
        logger.debug("ViT model is created.")
        # extract ViT model parameters into a variable
        target_state_dict = target_model.state_dict()

        # since names of layers are different between our implementation and the one from Huggingface,
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
            # So for now skip corresponding weights - they will be loaded later
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
                target_state_dict[target_key].copy_(source_weights)

        logger.debug("Weights are copied.")

        return target_model
