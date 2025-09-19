import torch
import torch.nn as nn
from typing import Iterable, List

class LoRALinear(nn.Module):
    """
    Drop-in wrapper for nn.Linear with additive LoRA branch.
    y = Wx + (alpha/r) * B(Ax)
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias is not None
        self.weight = base_linear.weight  # frozen by optimizer, kept here for state_dict compatibility
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        if r > 0:
            # LoRA factors (init: A zero, B zero except small init on A)
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = self.alpha / self.r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.base_linear.bias)
        if self.r > 0:
            lora = self.lora_B(self.lora_A(x)) * self.scaling
            return base + lora
        return base

def _matches_any(name: str, needles: Iterable[str]) -> bool:
    return any(needle in name for needle in needles)

def inject_lora(model: nn.Module, target_substrings: List[str], r: int = 8, alpha: int = 16) -> List[str]:
    """
    Recursively replace nn.Linear in modules whose names include any of target_substrings.
    Returns list of replaced module names (for logging).
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_any(name, target_substrings):
            # find parent to set attribute
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            # walk to parent
            parent = model
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)
            device = module.weight.device
            dtype  = module.weight.dtype        
            lora_lin = LoRALinear(module, r=r, alpha=alpha).to(device=device, dtype=dtype)
            setattr(parent, attr_name, lora_lin)
            replaced.append(name)
    return replaced

def lora_parameters(module: nn.Module):
    """Yield only LoRA parameters to optimize."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()
