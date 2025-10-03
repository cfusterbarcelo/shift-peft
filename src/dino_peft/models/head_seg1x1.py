import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    """
    2× upsample via ConvTranspose2d (k=4,s=2,p=1) + conv refinement.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(out_ch)
        self.act1   = nn.GELU()
        self.conv   = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(out_ch)
        self.act2   = nn.GELU()

    def forward(self, x):
        x = self.deconv(x)   # H,W ×2
        x = self.act1(self.bn1(x))
        x = self.conv(x)
        x = self.act2(self.bn2(x))
        return x

class SegHeadDeconv(nn.Module):
    """
    Pyramid decoder:
      - stem 1×1 to shrink ViT channels
      - 3–4× UpBlock (2× each): 25→50→100→200→400
      - final 1×1 to logits (K) at ~400×400
      - interpolate to out_hw (e.g., 350×350)
    """
    def __init__(self, in_ch: int, num_classes: int, n_ups: int = 4, base_ch: int = 512):
        super().__init__()
        assert n_ups in (3,4), "n_ups should be 3 or 4"
        # shrink ViT channels first
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )
        chs = [base_ch, base_ch//2, base_ch//4, base_ch//8, max(base_ch//8, 32)]
        ups = []
        for i in range(n_ups):
            ups.append(UpBlock(chs[i], chs[i+1]))
        self.ups = nn.Sequential(*ups)

        self.head = nn.Conv2d(chs[n_ups], num_classes, kernel_size=1, bias=True)

    def forward(self, feats, out_hw):
        x = self.stem(feats)          # (B, base_ch, 25, 25)
        x = self.ups(x)               # (B, chs[-1], 25*2^n, 25*2^n) → 200 or 400
        logits = self.head(x)         # (B, K, ~200|400, ~200|400)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits
