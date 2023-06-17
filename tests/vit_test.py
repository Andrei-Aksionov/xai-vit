import torch

from src.models.vit.embeddings import PatchEmbeddings
from src.models.vit.vision_transformer import ViT
from tests.config import num_tests


class TestViT:
    # TODO: get rid of constants
    def test_output_shape(self) -> None:
        # Given
        x = torch.rand((1, 3, 224, 224))
        B, C, H, W = x.shape
        num_classes = 1_000
        # TODO: the model should do patch embeddings
        embeddings = PatchEmbeddings(16)
        model = ViT(
            embeddings_size=128,
            context_size=1240,
            head_size=None,
            num_heads=2,
            feed_forward_scaling=4,
            num_layers=2,
            bias=False,
            dropout=0.0,
            num_classes=num_classes,
            num_channels=3,
            patch_size=16,
        )

        # When
        x = embeddings(x)
        out = model(x)

        # Then
        assert out.shape == (B, num_classes)
