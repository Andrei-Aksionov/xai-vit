#  Derived from https://github.com/jacobgil/vit-explain
#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2020 Jacob Gildenblat
#  Licensed under the MIT License (MIT).
#  ------------------------------------------------------------------------------------------

# flake8: noqa


import numpy as np
import torch


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for idx, attention in enumerate(attentions):
            # if idx < len(attentions) - 1:
            #     continue
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                # NOTE: min and max returns values and indices
                # attention_heads_fused = attention.max(axis=1)[0]  # (B, nh, T, T)
                attention_heads_fused = attention.max(axis=1).values  # (B, T, T) (1, 197, 197)
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1).values
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)  # (B, T * T) (1, 38809)
            _, indices = flat.topk(
                int(flat.size(-1) * discard_ratio), -1, False
            )  # (k, dim, largest) return top 90%, False - return smallest
            # TODO: why do we need it?
            # NOTE: this is done to preserve the first token - CLS
            indices = indices[indices != 0]
            # flat[0, indices] = 0
            flat[:, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            # TODO: why do we divide by 2?
            # a = (attention_heads_fused + 1.0 * I) / 2
            a = attention_heads_fused + I
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)  # (B, T, T)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]  # (T - 1)
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(
        self,
        model,
        attention_layer_name="attn_drop",
        head_fusion="mean",
        discard_ratio=0.9,
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
            if "attn" in name:
                module.fused_attn = False

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
