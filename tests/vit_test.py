from urllib.request import urlopen

import pytest
import torch
from PIL import Image
from transformers import ViTForImageClassification

from src.data.transform import ImageTransform
from src.models.vit.embeddings import PatchEmbeddings
from src.models.vit.vision_transformer import ViT
from tests.config import num_tests


class TestViT:
    @classmethod
    def setup_class(cls: "TestViT") -> None:
        cls.model_name = "google/vit-base-patch16-224"
        cls.hf_model = ViTForImageClassification.from_pretrained(cls.model_name)
        cls.vit_model = ViT.from_pretrained(cls.model_name)

    # TODO: get rid of constants
    def test_output_shape(self) -> None:
        # Given
        x = torch.rand((1, 3, 224, 224))
        B, C, H, W = x.shape
        num_classes = 1_000

        model = ViT(
            embeddings_size=128,
            head_size=None,
            num_heads=2,
            feed_forward_scaling=4,
            num_layers=2,
            bias=False,
            dropout=0.0,
            num_classes=num_classes,
            num_channels=3,
            patch_size=16,
            image_size=224,
        )

        # When
        out = model(x)

        # Then
        assert out.shape == (B, num_classes)

    @pytest.mark.parametrize(
        "image_url",
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
        ),
    )
    def test_predictions_with_huggingface_model(self, image_url: str) -> None:
        # Given
        transform = ImageTransform()
        image = Image.open(urlopen(image_url))
        image = transform(image)

        # When
        hs_prediction = TestViT.hf_model(image).logits.argmax(dim=-1).item()
        our_prediction = TestViT.vit_model(image).argmax(dim=-1).item()

        # Then
        assert hs_prediction == our_prediction
