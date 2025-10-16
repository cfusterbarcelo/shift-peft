import torch

@torch.no_grad()
def colorize_mask(m: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    m: (B, H, W) int tensor of class ids
    returns: (B, 3, H, W) float32 in [0,1]
    """
    B, H, W = m.shape
    out = torch.zeros(B, 3, H, W, device=m.device, dtype=torch.float32)

    if num_classes == 2:
        fg = (m == 1).float().unsqueeze(1)    # (B,1,H,W)
        out = out + fg                        # white for class 1, black for 0
        return out

    # multi-class palette (extend if you need more)
    palette = torch.tensor([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
        [1,0,1], [0,1,1], [1,0.5,0], [0.5,0,1], [0.5,0.5,0.5]
    ], device=m.device, dtype=torch.float32)

    for k in range(min(num_classes, palette.shape[0])):
        maskk = (m == k).float().unsqueeze(1)          # (B,1,H,W)
        out += maskk * palette[k].view(1,3,1,1)
    return out.clamp_(0, 1)