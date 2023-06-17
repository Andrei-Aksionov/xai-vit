import einops
from torch import Tensor


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
