# scripts/extract_features.py
# Script to extract features from images in a folder using DINOv2 and save them to a .npz file.

from pathlib import Path

import numpy as np
import torch
import yaml
import sys

from dino_peft.analysis.feature_extractor import extract_features_from_folder

DEFAULT_CFG_PATH = Path(__file__).parent.parent / "config" / "em_unsupervised_features_mac.yaml"

def load_config(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {cfg_path} is empty or invalid.")
    return cfg

def resolve_device(device_str: str) -> str:
    """
    Resolve device string into one of: 'cuda', 'mps', 'cpu'.

    - 'auto': prefer CUDA if available, then MPS (Apple Silicon), else CPU.
    - 'cuda' / 'mps' / 'cpu': force that device, with a sanity check.
    """
    device_str = device_str.lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but torch.cuda.is_available() is False.")
        return "cuda"

    if device_str == "mps":
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            raise RuntimeError("Requested device 'mps' but MPS backend is not available.")
        return "mps"

    if device_str == "cpu":
        return "cpu"

    raise ValueError(f"Invalid device '{device_str}', expected 'auto', 'cpu', 'cuda', or 'mps'.")

def main() -> None:
    cfg_path = DEFAULT_CFG_PATH
    print(f"[extract_features] Using config file: {cfg_path}")

    cfg = load_config(cfg_path)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    img_size = int(runtime_cfg.get("img_size", 518))


    data_dir = data_cfg.get("data_dir")
    output_path = data_cfg.get("output_path")

    if data_dir is None or output_path is None:
        raise ValueError(
            "Config must define data.data_dir and data.output_path.\n"
            f"Got data section: {data_cfg}"
        )

    dino_size = model_cfg.get("dino_size", "base")

    batch_size = int(runtime_cfg.get("batch_size", 16))
    num_workers = int(runtime_cfg.get("num_workers", 4))
    device_str = runtime_cfg.get("device", "auto")
    device = resolve_device(device_str)

    print(f"[extract_features] data_dir   = {data_dir}")
    print(f"[extract_features] output     = {output_path}")
    print(f"[extract_features] dino_size  = {dino_size}")
    print(f"[extract_features] batch_size = {batch_size}")
    print(f"[extract_features] num_workers= {num_workers}")
    print(f"[extract_features] device     = {device}")

    features_dict = extract_features_from_folder(
        data_dir=data_dir,
        dino_size=dino_size,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure everything is NumPy / Python-native before saving
    to_save = {}
    for k, v in features_dict.items():
        if isinstance(v, torch.Tensor):
            to_save[k] = v.detach().cpu().numpy()
        else:
            to_save[k] = v

    np.savez_compressed(out_path, **to_save)

    feats = features_dict["features"]
    print(
        f"[extract_features] Saved features to {out_path} "
        f"(N={feats.shape[0]}, dim={feats.shape[1]})"
    )
    print(f"[extract_features] Keys: {list(features_dict.keys())}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[extract_features] ERROR: {e}", file=sys.stderr)
        raise