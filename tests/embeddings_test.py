import random

import pytest
import torch

from src.models.vit.embeddings import PatchEmbeddings
from tests.config import num_tests


class TestPatchEmbeddings:
    @pytest.mark.parametrize(
        ("batch_size", "channels", "image_size", "patch_size"),
        list(
            zip(
                [random.randint(1, 10) for _ in range(num_tests)],
                [random.randint(1, 5) for _ in range(num_tests)],
                *(zip(*[(patch * random.randint(1, 11), patch) for patch in random.sample(range(1, 30), num_tests)])),
            ),
        ),
    )
    def test_return_patches(self, batch_size: int, channels: int, image_size: int, patch_size: int) -> None:
        # Given
        x = torch.randn((batch_size, channels, image_size, image_size))
        B, C, H, W = x.shape
        P = patch_size
        embeddings = PatchEmbeddings(P)

        # When
        out = embeddings(x)

        # Then
        assert out.shape == (B, (H // P) * (W // P), C * P * P)
