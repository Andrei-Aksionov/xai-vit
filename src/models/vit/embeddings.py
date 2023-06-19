import einops
import torch
from torch import Tensor, nn


class PatchEmbeddings:
    def __init__(self, patch_size: int) -> None:
        """Convert input image (B, C, H, W) into a set of patch embeddings - (B, num_patches^2, embeddings_size).

        Parameters
        ----------
        patch_size : int
            size of the patch (square size)
        """
        self.patch_size = patch_size

    def __call__(self, x: Tensor, use_einops: bool = True) -> Tensor:
        """Calculate how many patches can be fitted into height and width of the input image,
        reshape the image and flatten it along number of patches and embeddings dimensions.

        Parameters
        ----------
        x : Tensor
            input image of shape (B, C, H, W)
        use_einops : bool, optional
            if True - einops library will be used, by default False

        Returns
        -------
        Tensor
            output tensor of shape (B, number of patches, embeddings dimension), where embeddings dimension -
            input number of channels (C) times squared size of the patch (P^2)
        """

        # Notation:
        # Ht, Wt - how many patches can be fitted into height/width dimension
        # P - patch size

        B, C, H, W = x.shape

        if use_einops:
            return einops.rearrange(
                x,
                "B C (Ht P1) (Wt P2) -> B (Ht Wt) (C P1 P2)",
                P1=self.patch_size,
                P2=self.patch_size,
            )  # (B, Ht*Wt, C*P*P)

        x = x.reshape(
            B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size
        )  # (B, C, Ht, P, Wt, P)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, Ht, Wt, C, P, P)
        x = x.flatten(1, 2)  # (B, Ht*Wt, C, P, P)
        x = x.flatten(2, 4)  # (B, Ht*Wt, C*P*P)
        return x

    def get_num_patches(self, image_size: int) -> int:
        return (image_size // self.patch_size) ** 2


class PatchEmbeddingsTimm(nn.Module):
    def __init__(self, patch_size, in_channels, embeddings_size, bias) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embeddings_size = embeddings_size
        self.bias = bias

        self.projection = nn.Conv2d(in_channels, embeddings_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.projection(x)  # (1, 768, 14, 14)
        # x = x.flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, L, C) (1, 196, 768)
        x = einops.rearrange(x, "B C kh kw -> B (kh kw) C")
        return x

    def get_num_patches(self, image_size: int) -> int:
        return (image_size // self.patch_size) ** 2


# TODO: remove it
if __name__ == "__main__":
    embeddings = PatchEmbeddingsTimm(patch_size=16, in_channels=3, embeddings_size=768, bias=True)
    x = torch.randn((1, 3, 224, 224))
    out = embeddings(x)
