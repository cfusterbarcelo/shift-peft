from __future__ import annotations

from pathlib import Path
import inspect
from typing import Optional

import torch
import torch.nn as nn

from .base import (
    BackboneAdapter,
    BackboneOutput,
    infer_grid_size,
    infer_num_register_tokens,
    infer_patch_size,
    split_special_tokens,
)

try:  # pragma: no cover - optional dependency
    import open_clip
except Exception as exc:  # pragma: no cover
    open_clip = None
    _OPENCLIP_IMPORT_ERR = exc
else:  # pragma: no cover
    _OPENCLIP_IMPORT_ERR = None


def _require_openclip() -> None:
    if open_clip is None:
        raise ImportError(
            "open_clip is not available. Install open-clip-torch to use the OpenCLIP backbone."
        ) from _OPENCLIP_IMPORT_ERR


def _resolve_pretrained_and_weights(
    pretrained: str | None,
    weights: str | None,
) -> tuple[str | None, str | None]:
    weights_path = None
    pretrained_tag = pretrained or None
    if pretrained:
        candidate = Path(pretrained).expanduser()
        if candidate.exists():
            weights_path = str(candidate)
            pretrained_tag = None
    if weights:
        candidate = Path(weights).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"OpenCLIP weights not found: {candidate}")
        weights_path = str(candidate)
    return pretrained_tag, weights_path


def _load_openclip_model(
    model_name: str,
    pretrained: str | None,
    weights: str | None,
    device: torch.device,
) -> nn.Module:
    _require_openclip()
    pretrained_tag, weights_path = _resolve_pretrained_and_weights(pretrained, weights)
    model = open_clip.create_model(model_name, pretrained=pretrained_tag)
    if weights_path:
        if hasattr(open_clip, "load_checkpoint"):
            open_clip.load_checkpoint(model, weights_path)
        else:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _infer_embed_dim(model: nn.Module) -> int:
    for attr in ("embed_dim", "width", "hidden_dim", "dim"):
        val = getattr(model, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    proj = getattr(model, "proj", None)
    if hasattr(proj, "shape") and len(proj.shape) >= 2:
        return int(proj.shape[0])
    raise ValueError("Unable to infer embed_dim from OpenCLIP visual model.")


def _extract_tokens(output: object) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor) and output.dim() == 3:
        return output
    if isinstance(output, dict):
        for key in (
            "x",
            "x_norm",
            "x_norm_patchtokens",
            "patch_tokens",
            "tokens",
            "last_hidden_state",
        ):
            tokens = output.get(key)
            if isinstance(tokens, torch.Tensor) and tokens.dim() == 3:
                return tokens
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
    return None


def _shape_tokens(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tokens is None:
        return None
    if tokens.dim() == 2:
        return None
    if tokens.dim() != 3:
        return None
    return tokens


class OpenCLIPAdapter(BackboneAdapter):
    def __init__(
        self,
        model_name: str,
        device: str | torch.device,
        *,
        pretrained: str | None = None,
        weights: str | None = None,
    ) -> None:
        super().__init__(name="openclip", variant=model_name)
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        clip_model = _load_openclip_model(model_name, pretrained, weights, device_obj)
        visual = getattr(clip_model, "visual", None)
        if visual is None:
            raise RuntimeError("OpenCLIP model does not expose a visual tower.")

        self.model = visual
        self.embed_dim = _infer_embed_dim(visual)
        self.patch_size = infer_patch_size(visual, fallback=14)
        self.num_register_tokens = infer_num_register_tokens(visual)
        self.to(device_obj)
        self.eval()

    def _maybe_resize_pos_embed(self, x: torch.Tensor) -> None:
        if not hasattr(self.model, "image_size"):
            return
        image_size = getattr(self.model, "image_size")
        if isinstance(image_size, (tuple, list)):
            if len(image_size) == 2:
                target = (int(image_size[0]), int(image_size[1]))
            else:
                target = (int(image_size[0]), int(image_size[0]))
        elif isinstance(image_size, int):
            target = (image_size, image_size)
        else:
            return
        current = (int(x.shape[-2]), int(x.shape[-1]))
        if current == target:
            return
        if hasattr(self.model, "set_image_size"):
            try:
                self.model.set_image_size(current)
                return
            except Exception as exc:
                raise RuntimeError(
                    "OpenCLIP visual model failed to resize positional embeddings via set_image_size."
                ) from exc
        if hasattr(self.model, "resize_pos_embed"):
            try:
                self.model.resize_pos_embed(current)
                return
            except Exception as exc:
                raise RuntimeError(
                    "OpenCLIP visual model failed to resize positional embeddings via resize_pos_embed."
                ) from exc
        raise RuntimeError(
            "OpenCLIP visual model does not support resizing positional embeddings "
            f"from {target} to {current}. Consider using the native image size."
        )

    def _get_tokens(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if hasattr(self.model, "forward_features"):
            try:
                fn = self.model.forward_features
                sig = inspect.signature(fn)
                kwargs = {}
                for key in (
                    "return_all_tokens",
                    "return_tokens",
                    "return_patch_tokens",
                    "output_tokens",
                    "return_all_features",
                ):
                    if key in sig.parameters:
                        kwargs[key] = True
                output = fn(x, **kwargs) if kwargs else fn(x)
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        if hasattr(self.model, "forward_intermediates"):
            try:
                fn = self.model.forward_intermediates
                sig = inspect.signature(fn)
                kwargs = {}
                for key in ("return_all_tokens", "return_tokens", "return_all_features"):
                    if key in sig.parameters:
                        kwargs[key] = True
                output = fn(x, **kwargs) if kwargs else fn(x)
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        if hasattr(self.model, "get_intermediate_layers"):
            try:
                output = self.model.get_intermediate_layers(x, n=1, reshape=False)
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        for flag in ("return_all_tokens", "return_tokens", "return_all_features", "output_tokens"):
            try:
                output = self.model(x, **{flag: True})
            except TypeError:
                continue
            tokens = _shape_tokens(_extract_tokens(output))
            if tokens is not None:
                return tokens

        tokens = self._manual_vit_tokens(x)
        if tokens is not None:
            return tokens
        return None

    def _manual_vit_tokens(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer")
        if not all(hasattr(self.model, attr) for attr in required):
            return None
        model = self.model
        try:
            x = model.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = model.class_embedding
            if cls.dim() == 1:
                cls = cls.unsqueeze(0)
            cls = cls.to(dtype=x.dtype, device=x.device)
            cls = cls.expand(x.shape[0], 1, -1)
            x = torch.cat([cls, x], dim=1)
            pos = model.positional_embedding
            if pos is not None:
                if pos.dim() == 2:
                    pos = pos.unsqueeze(0)
                if pos.shape[1] == x.shape[1]:
                    x = x + pos.to(dtype=x.dtype, device=x.device)
            x = model.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            if hasattr(model, "ln_post"):
                x = model.ln_post(x)
            return _shape_tokens(x)
        except Exception:
            return None

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        self._maybe_resize_pos_embed(x)
        grid_h, grid_w = infer_grid_size(x, self.patch_size)
        num_patches = grid_h * grid_w

        tokens = self._get_tokens(x)
        if tokens is None:
            raise RuntimeError("Unable to extract tokens from OpenCLIP visual backbone.")

        try:
            patch_tokens, cls_token, register_tokens = split_special_tokens(
                tokens,
                num_patches=num_patches,
                num_register_tokens=self.num_register_tokens,
            )
        except ValueError as exc:
            raise RuntimeError(
                "OpenCLIP token count mismatch. This usually means positional embedding "
                "interpolation is unsupported for the requested image size."
            ) from exc

        if cls_token is not None and cls_token.numel() > 0:
            global_embedding = cls_token.squeeze(1)
        else:
            global_embedding = patch_tokens.mean(dim=1)
        return BackboneOutput(
            global_embedding=global_embedding,
            patch_tokens=patch_tokens,
            grid_size=(grid_h, grid_w),
            cls_token=cls_token,
            register_tokens=register_tokens,
        )
