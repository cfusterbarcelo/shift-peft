import torch.nn as nn
import torch.nn.functional as F

class SegHead1x1(nn.Module):
    """
    Simple 1×1 conv to num_classes + bilinear upsample to input size.
    Expects backbone features (B, C, H', W').
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, feats, out_hw):
        logits_low = self.proj(feats)  # (B, K, H', W')
        return F.interpolate(logits_low, size=out_hw, mode="bilinear", align_corners=False)
