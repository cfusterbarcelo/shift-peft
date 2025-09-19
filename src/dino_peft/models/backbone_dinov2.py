import torch
import torch.nn as nn

NAME_MAP = {
    "small": "dinov2_vits14_reg",
    "base":  "dinov2_vitb14_reg",
    "large": "dinov2_vitl14_reg",
    "giant": "dinov2_vitg14_reg",
}
PATCH_SIZE = 14

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, size="base", device="cuda"):
        super().__init__()
        name = NAME_MAP[size]
        self.vit = torch.hub.load('facebookresearch/dinov2', name)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = PATCH_SIZE
        self.to(device)
        self.eval()

    @torch.no_grad()
    def infer_grid_hw(self, img_h, img_w):
        return img_h // self.patch_size, img_w // self.patch_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns features as (B, C, H', W') robustly, regardless of how the hub returns shape.
        """
        B, _, H, W = x.shape
        Hh, Ww = self.infer_grid_hw(H, W)

        # Try reshape=True first
        out = self.vit.get_intermediate_layers(x, n=1, reshape=True)[0]
        # Cases we’ve seen in the wild:
        #  (B, H', W', C)  or  (B, H', C, W')
        if out.dim() == 4:
            if out.shape[-1] == self.embed_dim:         # (B, H', W', C)
                feats = out.permute(0, 3, 1, 2).contiguous()
                return feats
            if out.shape[2] == self.embed_dim:          # (B, H', C, W')
                feats = out.permute(0, 2, 1, 3).contiguous()
                return feats

        # Fallback: reshape=False → (B, N, C), possibly with/without CLS
        out_tokens = self.vit.get_intermediate_layers(x, n=1, reshape=False)[0]  # (B, N, C)
        N = out_tokens.shape[1]
        # If there is a CLS token (N == H'*W' + 1), drop it
        expect_N = Hh * Ww
        if N == expect_N + 1:
            out_tokens = out_tokens[:, 1:, :]  # drop CLS
        elif N != expect_N:
            # As a last resort, try to infer H'W' from N (no CLS)
            expect_N = N
        # (B, N, C) -> (B, C, H', W')
        feats = out_tokens.transpose(1, 2).contiguous().reshape(B, self.embed_dim, Hh, Ww)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
