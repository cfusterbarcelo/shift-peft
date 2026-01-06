from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Drop-in wrapper for nn.Linear with additive LoRA branch.
    y = Wx + (alpha/r) * B(Ax)
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
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
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else None
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
            residual = self.dropout(x) if self.dropout is not None else x
            lora = self.lora_B(self.lora_A(residual)) * self.scaling
            return base + lora
        return base

def _matches_any(name: str, needles: Iterable[str]) -> bool:
    return any(needle in name for needle in needles)

def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    node: nn.Module = model
    for part in name.split("."):
        node = getattr(node, part)
    return node


def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name = ".".join(name.split(".")[:-1])
    attr_name = name.split(".")[-1]
    parent = model if not parent_name else _get_module_by_name(model, parent_name)
    setattr(parent, attr_name, new_module)


def inject_lora_by_names(
    model: nn.Module,
    target_names: List[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> List[str]:
    replaced: List[str] = []
    for name in target_names:
        module = _get_module_by_name(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target '{name}' is not an nn.Linear (got {type(module)}).")
        device = module.weight.device
        dtype = module.weight.dtype
        lora_lin = LoRALinear(module, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
        _replace_module(model, name, lora_lin)
        replaced.append(name)
    return replaced


def inject_lora(model: nn.Module, target_substrings: List[str], r: int = 8, alpha: int = 16) -> List[str]:
    """
    Legacy substring-based injector. Prefer apply_peft()/inject_lora_by_names().
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_any(name, target_substrings):
            device = module.weight.device
            dtype = module.weight.dtype
            lora_lin = LoRALinear(module, r=r, alpha=alpha).to(device=device, dtype=dtype)
            _replace_module(model, name, lora_lin)
            replaced.append(name)
    return replaced

def lora_parameters(module: nn.Module):
    """Yield only LoRA parameters to optimize."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            if m.lora_A is not None:
                yield from m.lora_A.parameters()
            if m.lora_B is not None:
                yield from m.lora_B.parameters()


@dataclass
class LoraConfig:
    enabled: bool
    target_policy: str
    layer_selection: str
    exclude: List[str]
    r: int
    alpha: int
    dropout: float
    compatibility_mode: bool


@dataclass
class LoraAudit:
    backbone_name: str
    backbone_variant: str
    policy: str
    rank: int
    alpha: int
    dropout: float
    compatibility_mode: bool
    layer_selection: str
    exclude: List[str]
    total_targets: int
    blocks_targeted: int
    block_count: int | None
    trainable_params: int
    lora_params: int
    targets: List[str]
    per_block: dict
    qkv_equivalence: bool


_BLOCK_RE = re.compile(r"(?:blocks|layers)\.(\d+)")


def _infer_block_count(model: nn.Module) -> int | None:
    for attr in ("blocks", "layers"):
        blocks = getattr(model, attr, None)
        if blocks is not None:
            try:
                return len(blocks)
            except TypeError:
                continue
    return None


def _block_index_from_name(name: str) -> int | None:
    match = _BLOCK_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def _classify_linear(name: str) -> str | None:
    lower = name.lower()
    if ".attn." in lower:
        if lower.endswith("qkv"):
            return "attn_qkv"
        if lower.endswith("q_proj"):
            return "attn_q_proj"
        if lower.endswith("k_proj"):
            return "attn_k_proj"
        if lower.endswith("v_proj"):
            return "attn_v_proj"
        if lower.endswith("out_proj") or lower.endswith("proj"):
            return "attn_proj"
    if ".mlp." in lower or ".ffn." in lower:
        if lower.endswith("fc1") or lower.endswith("w1") or lower.endswith("up_proj"):
            return "mlp_fc1"
        if lower.endswith("fc2") or lower.endswith("w2") or lower.endswith("down_proj"):
            return "mlp_fc2"
    return None


def resolve_lora_cfg(cfg: Mapping) -> LoraConfig:
    lora_cfg = dict(cfg.get("lora") or {}) if isinstance(cfg, Mapping) else {}
    enabled = lora_cfg.get("enabled")
    if enabled is None:
        enabled = cfg.get("use_lora", cfg.get("enable_lora", True))
    target_policy = lora_cfg.get("target_policy")
    legacy_targets = cfg.get("lora_targets") if isinstance(cfg, Mapping) else None
    if target_policy is None:
        if legacy_targets:
            if any("mlp" in t or "fc" in t for t in legacy_targets):
                target_policy = "vit_attention_mlp"
            else:
                target_policy = "vit_attention_only"
        else:
            target_policy = "vit_attention_only"
    target_policy = str(target_policy).lower()
    layer_selection = str(lora_cfg.get("layer_selection", "all")).lower()
    exclude = list(lora_cfg.get("exclude") or [])
    if not exclude:
        exclude = ["head", "decoder", "seg_head"]
    r = int(lora_cfg.get("rank", cfg.get("lora_rank", 8)))
    alpha = int(lora_cfg.get("alpha", cfg.get("lora_alpha", 16)))
    dropout = float(lora_cfg.get("dropout", cfg.get("lora_dropout", 0.0)))
    compatibility_mode = bool(lora_cfg.get("compatibility_mode", True))
    return LoraConfig(
        enabled=bool(enabled),
        target_policy=str(target_policy),
        layer_selection=str(layer_selection),
        exclude=exclude,
        r=r,
        alpha=alpha,
        dropout=dropout,
        compatibility_mode=compatibility_mode,
    )


def discover_lora_targets(model: nn.Module, lora_cfg: LoraConfig) -> tuple[list[str], dict, bool]:
    candidates = []
    linear_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        linear_names.append(name)
        if _matches_any(name, lora_cfg.exclude):
            continue
        block_idx = _block_index_from_name(name)
        if block_idx is None:
            continue
        kind = _classify_linear(name)
        if kind is None:
            continue
        candidates.append((name, kind, block_idx))

    if not candidates:
        sample = [n for n in linear_names if "attn" in n.lower() or "mlp" in n.lower()][:30]
        hint = "\n  ".join(sample) if sample else "(no candidate linear modules found)"
        raise RuntimeError(
            "LoRA target discovery found zero matching modules. "
            "Check backbone naming or update exclude list.\n"
            f"Candidate modules:\n  {hint}"
        )

    per_block: dict[int, dict[str, list[str]]] = {}
    for name, kind, block_idx in candidates:
        per_block.setdefault(block_idx, {}).setdefault(kind, []).append(name)

    targets: list[str] = []
    summary: dict[str, dict] = {}
    qkv_equivalence = False

    include_mlp = lora_cfg.target_policy == "vit_attention_mlp"
    for block_idx in sorted(per_block.keys()):
        kinds = per_block[block_idx]
        block_summary: dict[str, list[str]] = {}

        qkv_names = kinds.get("attn_qkv", [])
        q_proj = kinds.get("attn_q_proj", [])
        k_proj = kinds.get("attn_k_proj", [])
        v_proj = kinds.get("attn_v_proj", [])
        if qkv_names:
            targets.extend(sorted(qkv_names))
            block_summary["qkv"] = sorted(qkv_names)
        elif q_proj or k_proj or v_proj:
            if lora_cfg.compatibility_mode and not (q_proj and k_proj and v_proj):
                raise RuntimeError(
                    "LoRA compatibility check failed: "
                    f"expected q_proj/k_proj/v_proj in block {block_idx}, "
                    f"found q={len(q_proj)} k={len(k_proj)} v={len(v_proj)}."
                )
            qkv_equivalence = True
            targets.extend(sorted(q_proj + k_proj + v_proj))
            block_summary["q_proj"] = sorted(q_proj)
            block_summary["k_proj"] = sorted(k_proj)
            block_summary["v_proj"] = sorted(v_proj)

        proj_names = kinds.get("attn_proj", [])
        if proj_names:
            targets.extend(sorted(proj_names))
            block_summary["proj"] = sorted(proj_names)

        if include_mlp:
            fc1_names = kinds.get("mlp_fc1", [])
            fc2_names = kinds.get("mlp_fc2", [])
            if fc1_names:
                targets.extend(sorted(fc1_names))
                block_summary["fc1"] = sorted(fc1_names)
            if fc2_names:
                targets.extend(sorted(fc2_names))
                block_summary["fc2"] = sorted(fc2_names)

        summary[str(block_idx)] = block_summary

    if not targets:
        raise RuntimeError(
            "LoRA target discovery produced no targets after applying policy. "
            "Verify that attention/MLP modules exist for this backbone."
        )
    return sorted(set(targets)), summary, qkv_equivalence


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _validate_trainable(model: nn.Module, allowed: Iterable[str] | None = None) -> None:
    allowed = list(allowed or [])
    offenders = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            continue
        if _matches_any(name, allowed):
            continue
        offenders.append(name)
    if offenders:
        raise RuntimeError(
            "Unexpected trainable backbone parameters after LoRA injection: "
            f"{offenders[:15]}{'...' if len(offenders) > 15 else ''}"
        )


def apply_peft(
    model: nn.Module,
    cfg: Mapping,
    run_dir: Path | None = None,
    backbone_info: Mapping | None = None,
    *,
    write_report: bool = True,
) -> LoraAudit | None:
    lora_cfg = resolve_lora_cfg(cfg)
    backbone_info = backbone_info or {}
    backbone_name = str(backbone_info.get("name", "unknown"))
    backbone_variant = str(backbone_info.get("variant", "unknown"))

    if lora_cfg.layer_selection != "all":
        raise ValueError(
            f"Unsupported lora.layer_selection '{lora_cfg.layer_selection}'. Only 'all' is supported."
        )
    if lora_cfg.target_policy not in ("vit_attention_only", "vit_attention_mlp"):
        raise ValueError(
            "Unsupported lora.target_policy "
            f"'{lora_cfg.target_policy}'. Use 'vit_attention_only' or 'vit_attention_mlp'."
        )

    if not lora_cfg.enabled:
        _freeze_module(model)
        _validate_trainable(model)
        return None

    targets, per_block, qkv_equivalence = discover_lora_targets(model, lora_cfg)
    replaced = inject_lora_by_names(
        model,
        targets,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
    )

    _freeze_module(model)
    for p in lora_parameters(model):
        p.requires_grad = True
    _validate_trainable(model)

    trainable_params = _count_trainable_params(model)
    lora_params = _count_lora_params(model)
    block_count = _infer_block_count(model)
    blocks_targeted = len(per_block)
    if lora_cfg.compatibility_mode and block_count is not None and blocks_targeted != block_count:
        raise RuntimeError(
            "LoRA compatibility check failed: "
            f"targeted {blocks_targeted} blocks but backbone has {block_count}."
        )

    audit = LoraAudit(
        backbone_name=backbone_name,
        backbone_variant=backbone_variant,
        policy=lora_cfg.target_policy,
        rank=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        compatibility_mode=lora_cfg.compatibility_mode,
        layer_selection=lora_cfg.layer_selection,
        exclude=lora_cfg.exclude,
        total_targets=len(replaced),
        blocks_targeted=blocks_targeted,
        block_count=block_count,
        trainable_params=trainable_params,
        lora_params=lora_params,
        targets=replaced,
        per_block=per_block,
        qkv_equivalence=qkv_equivalence,
    )

    print(
        "[lora] policy="
        f"{audit.policy} targets={audit.total_targets} blocks={audit.blocks_targeted}"
        f" trainable={audit.trainable_params:,} lora_params={audit.lora_params:,}"
    )
    if qkv_equivalence:
        print("[lora] qkv_equivalence: used q_proj/k_proj/v_proj in place of qkv.")

    if run_dir and write_report:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "lora_targets.json"
        report_path.write_text(json.dumps(asdict(audit), indent=2, sort_keys=True))

    return audit
