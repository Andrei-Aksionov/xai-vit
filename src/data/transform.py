from PIL import Image
from torch import Tensor
from torchvision import transforms


class ImageTransform:
    def __init__(self, mean: list[float] | None = None, std: list[float] | None = None) -> None:
        # TODO: this should be passed from config file
        # https://huggingface.co/google/vit-base-patch16-224
        # self.mean = mean or [0.49139968, 0.48215841, 0.44653091]
        # self.std = std or [0.24703223, 0.24348513, 0.26158784]
        # TODO: grab it from ViTImageProcessor
        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def __call__(self, x: Image) -> Tensor:
        x = self.transform(x)
        if x.ndim == 3:
            return x[None, :]
