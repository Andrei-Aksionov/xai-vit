import random

import pytest
import torch

from src import config
from src.models.vit.embeddings import ImageToPatches, PatchEmbeddings

NUM_TESTS = config.tests.num_tests


class TestImageToPatches:
    @pytest.mark.parametrize(
        ("batch_size", "channels", "image_size", "patch_size"),
        list(
            zip(
                [random.randint(1, 10) for _ in range(NUM_TESTS)],
                [random.randint(1, 5) for _ in range(NUM_TESTS)],
                *(zip(*[(patch * random.randint(1, 11), patch) for patch in random.sample(range(1, 30), NUM_TESTS)])),
            ),
        ),
    )
    def test_return_patches(self, batch_size: int, channels: int, image_size: int, patch_size: int) -> None:
        # Given
        x = torch.randn((batch_size, channels, image_size, image_size))
        B, C, H, W = x.shape
        P = patch_size
        embeddings = ImageToPatches(P)

        # When
        out = embeddings(x)

        # Then
        assert out.shape == (B, (H // P) * (W // P), C * P * P)


class TestPatchEmbeddings:
    @pytest.mark.parametrize(
        ("batch_size", "channels", "image_size", "patch_size", "embeddings_size"),
        list(
            zip(
                [random.randint(1, 10) for _ in range(NUM_TESTS)],
                [random.randint(1, 5) for _ in range(NUM_TESTS)],
                *(zip(*[(patch * random.randint(1, 11), patch) for patch in random.sample(range(1, 30), NUM_TESTS)])),
                [random.randint(1, 256) for _ in range(NUM_TESTS)],
            ),
        ),
    )
    def test_return_patches(
        self,
        batch_size: int,
        channels: int,
        image_size: int,
        patch_size: int,
        embeddings_size: int,
    ) -> None:
        # Given
        x = torch.randn((batch_size, channels, image_size, image_size))
        B, C, H, W = x.shape
        P = patch_size
        embeddings = PatchEmbeddings(P, C, embeddings_size)

        # When
        out = embeddings(x)

        # Then
        assert out.shape == (B, (H // P) * (W // P), embeddings_size)
