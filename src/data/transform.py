from PIL import Image
from torch import Tensor
from torchvision import transforms


class ImageTransform:
    def __init__(
        self,
        resize_to: int | tuple[int, int] = (224, 224),
        mean: tuple[float] = (0.5, 0.5, 0.5),
        std: tuple[float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Transform input image into a tensor.

        Transformation includes resizing, conversion to pytorch tensor and normalization.

        Parameters
        ----------
        resize_to : int | tuple[int, int], optional
            to what size the input image should be resized, by default (224, 224)
        mean : tuple[float], optional
            mean values of pixels for each channel, by default (0.5, 0.5, 0.5)
        std : tuple[float], optional
            standard deviation of pixel values for each channel, by default (0.5, 0.5, 0.5)
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_to),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ],
        )

    def __call__(self, x: Image) -> Tensor:  # noqa: D102
        x = self.transform(x)
        # if it's a single image - add artificial batch size of 1
        return x[None, :] if x.ndim == 3 else x

    # TODO: make a method that loads config from Huggingface model
