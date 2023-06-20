import einops
from torch import Tensor, nn


class ImageToPatches:
    def __init__(self, patch_size: int) -> None:
        """View the input image (B, C, H, W) as a set of patches (B, num_patches^2, embeddings_size).

        Parameters
        ----------
        patch_size : int
            size of the patch (square size)
        """
        self.patch_size = patch_size

    def __call__(self, x: Tensor, use_einops: bool = True) -> Tensor:
        """View input tensor `x` as a set of patches.

        Calculate how many patches can be fitted into height and width of the input tensor, reshape the tensor and
        flatten it along number of patches and embeddings dimensions.

        Parameters
        ----------
        x : Tensor
            input tensor of shape (B, C, H, W)
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

        B, C, H, W = x.shape  # noqa: N806
        P = self.patch_size  # noqa: N806

        if use_einops:
            return einops.rearrange(
                x,
                "B C (Ht P1) (Wt P2) -> B (Ht Wt) (C P1 P2)",
                P1=P,
                P2=P,
            )  # (B, Ht*Wt, C*P*P)

        x = x.reshape(B, C, H // P, P, W // P, P)  # (B, C, Ht, P, Wt, P)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, Ht, Wt, C, P, P)
        x = x.flatten(1, 2)  # (B, Ht*Wt, C, P, P)
        return x.flatten(2, 4)  # (B, Ht*Wt, C*P*P)

    def get_num_patches(self, image_size: int) -> int:
        """Given image size and patch size calculate how many patches can be fitted.

        Parameters
        ----------
        image_size : int
            size of the image. For now only squared images are supported.

        Returns
        -------
        int
            number of patches that can be fitted in the image of provided size.
        """
        return (image_size // self.patch_size) ** 2


class PatchEmbeddings(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embeddings_size: int) -> None:
        """Transform input tensor into a set of patches that have embeddings size mapped to the desired one.

        Parameters
        ----------
        patch_size : int
            size of the patch
        in_channels : int
            number of channels of expected input tensor
        embeddings_size : int
            desired output embeddings size of each patch
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embeddings_size = embeddings_size

        self.projection = nn.Conv2d(in_channels, embeddings_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Transform input tensor `x` into set of patches that has desired embeddings size.

        Parameters
        ----------
        x : Tensor
            input tensor of shape (B, C, H, W)

        Returns
        -------
        Tensor
            output tensor of shape (B, number of patches, embeddings dimension)
        """
        # Notation:
        # fmH, fmW - feature map height and width correspondingly
        # embd - embeddings size

        B, C, H, W = x.shape  # noqa: N806
        x = self.projection(x)  # (B, embeddings_size, fmH, fmW)
        return einops.rearrange(x, "B embd fmH fmW -> B (fmH fmW) embd")

    def get_num_patches(self, image_size: int) -> int:
        """Given image size and patch size calculate how many patches can be fitted.

        Parameters
        ----------
        image_size : int
            size of the image. For now only squared images are supported.

        Returns
        -------
        int
            number of patches that can be fitted in the image of provided size.
        """
        return (image_size // self.patch_size) ** 2
