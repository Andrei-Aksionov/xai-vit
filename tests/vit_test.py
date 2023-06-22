import random
from urllib.request import urlopen

import pytest
import torch
from PIL import Image
from transformers import ViTForImageClassification

from src import config
from src.data.transform import ImageTransform
from src.models.vit.vision_transformer import ViT

NUM_TESTS = config.tests.num_tests


class TestViT:
    @classmethod
    def setup_class(cls: "TestViT") -> None:
        # models loading is done once for all tests
        cls.model_name = config.model.vit.pretrained.name
        cls.hf_model = ViTForImageClassification.from_pretrained(cls.model_name)
        cls.vit_model = ViT.from_pretrained(cls.model_name)

    @pytest.mark.parametrize(
        (
            "batch_size",
            "num_channels",
            "image_size",
            "patch_size",
            "embeddings_size",
            "num_heads",
            "num_classes",
            "feed_forward_scaling",
            "num_layers",
            "bias",
            "dropout",
        ),
        list(
            zip(
                [random.randint(1, 10) for _ in range(NUM_TESTS)],  # batch_size
                [random.randint(1, 5) for _ in range(NUM_TESTS)],  # num_channels
                *(
                    zip(*[(ps * random.choice(range(1, 11)), ps) for ps in random.sample(range(1, 30), NUM_TESTS)])
                ),  # image_size, patch_size
                *(
                    zip(
                        *[(hs * random.choice(range(1, 8)), hs) for hs in random.sample(range(1, 16), NUM_TESTS)],
                    )
                ),  # embeddings_size, num_heads
                [random.randint(1, 1_000) for _ in range(NUM_TESTS)],  # num_classes
                [random.randint(1, 4) for _ in range(NUM_TESTS)],  # feed_forward_scaling
                [random.randint(1, 4) for _ in range(NUM_TESTS)],  # num_layers
                [random.choice((True, False)) for _ in range(NUM_TESTS)],  # bias
                [random.random() for _ in range(NUM_TESTS)],  # dropout
            ),
        ),
    )
    def test_output_shape(
        self,
        batch_size: int,
        num_channels: int,
        image_size: int,
        patch_size: int,
        embeddings_size: int,
        num_heads: int,
        num_classes: int,
        feed_forward_scaling: int,
        num_layers: int,
        bias: bool,
        dropout: float,
    ) -> None:
        # Given
        x = torch.rand((batch_size, num_channels, image_size, image_size))
        B, C, H, W = x.shape

        model = ViT(
            embeddings_size=embeddings_size,
            head_size=None,
            num_heads=num_heads,
            feed_forward_scaling=feed_forward_scaling,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            num_classes=num_classes,
            num_channels=num_channels,
            patch_size=patch_size,
            image_size=image_size,
        )

        # When
        out = model(x)

        # Then
        assert out.shape == (B, num_classes)

    @pytest.mark.parametrize(
        "image_url",
        [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
        ],
    )
    def test_compare_predictions_with_huggingface_model(self, image_url: str) -> None:
        # Given
        transform = ImageTransform()
        image = Image.open(urlopen(image_url))
        image = transform(image)

        # When
        hs_prediction = TestViT.hf_model(image).logits.argmax(dim=-1).item()
        our_prediction = TestViT.vit_model(image).argmax(dim=-1).item()

        # Then
        assert hs_prediction == our_prediction

    @pytest.mark.slow
    def test_compare_predictions_with_huggingface_model_from_dataset(self) -> None:
        # Given
        from datasets import load_dataset

        dataset_config = config.tests.vit.dataset.huggingface
        dataset = load_dataset(dataset_config.name, split=dataset_config.split)
        transform = ImageTransform()

        # When
        hf_predictions, our_predictions = [], []
        for record in dataset:
            image = record["image"]
            # models are trained only on RGB images
            if image.mode != "RGB":
                continue
            image = transform(image)
            hf_predictions.append(TestViT.hf_model(image).logits.argmax(dim=-1).item())
            our_predictions.append(TestViT.vit_model(image).argmax(dim=-1).item())

        # Then
        assert hf_predictions == our_predictions
