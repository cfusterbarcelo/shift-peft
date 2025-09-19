from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.lora import inject_lora, lora_parameters
from dino_peft.models.head_seg1x1 import SegHead1x1


def pick_device(cfg_device: str | None):
    if cfg_device and cfg_device.lower() != "auto":
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SegTrainer:
    def __init__(self, cfg_path: str):
        # -------- config & device ----------
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = pick_device(self.cfg.get("device", "auto"))
        print(">> Using device:", self.device)

        self.out_dir = Path(self.cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # -------- transforms ----------
        t = em_seg_transforms(tuple(self.cfg["img_size"]))

        # -------- datasets (single train/val pair) ----------
        if "train_img_dir" not in self.cfg or "train_mask_dir" not in self.cfg:
            raise ValueError("Config must provide train_img_dir and train_mask_dir.")

        self.train_ds = PairedDirsSegDataset(
            self.cfg["train_img_dir"],
            self.cfg["train_mask_dir"],
            img_size=self.cfg["img_size"],
            to_rgb=True,
            transform=t,
            binarize=bool(self.cfg.get("binarize", True)),
            binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
        )

        if "val_img_dir" in self.cfg and "val_mask_dir" in self.cfg:
            self.val_ds = PairedDirsSegDataset(
                self.cfg["val_img_dir"],
                self.cfg["val_mask_dir"],
                img_size=self.cfg["img_size"],
                to_rgb=True,
                transform=t,
                binarize=bool(self.cfg.get("binarize", True)),
                binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
            )
        else:
            # optional fallback: split a bit from train if no explicit val set provided
            val_ratio = float(self.cfg.get("val_ratio", 0.1))
            n = len(self.train_ds)
            n_val = max(1, int(round(n * val_ratio)))
            n_train = n - n_val
            g = torch.Generator().manual_seed(int(self.cfg.get("split_seed", 42)))
            self.train_ds, self.val_ds = random_split(self.train_ds, [n_train, n_val], generator=g)

        # -------- loaders ----------
        pin = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
        )

        # -------- model ----------
        self.backbone = DINOv2FeatureExtractor(size=self.cfg["dino_size"], device=str(self.device))
        in_ch = self.backbone.embed_dim
        self.head = SegHead1x1(in_ch, self.cfg["num_classes"]).to(self.device)

        # -------- LoRA ----------
        self.lora_names = []
        if self.cfg.get("use_lora", True):
            self.lora_names = inject_lora(
                self.backbone.vit,
                target_substrings=self.cfg.get("lora_targets", ["attn.qkv", "attn.proj"]),
                r=int(self.cfg.get("lora_rank", 8)),
                alpha=int(self.cfg.get("lora_alpha", 16)),
            )
        # ensure any new LoRA modules are on our device
        self.backbone.to(self.device)

        # freeze base, enable LoRA + head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in lora_parameters(self.backbone):
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        # -------- optim / loss ----------
        params = list(self.head.parameters()) + list(lora_parameters(self.backbone))
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = int(self.cfg["epochs"])

        # -------- AMP (device-aware) ----------
        self.use_amp = bool(self.cfg.get("amp", True))
        if self.device.type == "cuda":
            from torch.amp import GradScaler, autocast
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
            self.autocast = lambda: autocast("cuda", enabled=self.use_amp)
        elif self.device.type == "mps":
            from torch import autocast
            self.scaler = None
            self.autocast = lambda: autocast(device_type="mps", enabled=self.use_amp)
        else:
            from torch import autocast
            self.scaler = None
            self.autocast = lambda: autocast(device_type="cpu", enabled=False)

        # -------- save config & lora list ----------
        with open(self.out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f)
        with open(self.out_dir / "lora_layers.txt", "w") as f:
            for n in self.lora_names:
                f.write(n + "\n")

    def _step(self, batch, train=True):
        imgs, masks = batch
        imgs = imgs.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        out_hw = masks.shape[-2:]
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            feats  = self.backbone(imgs)               # (B, C, H', W')
            logits = self.head(feats, out_hw)          # (B, K, H, W)
            loss   = self.criterion(logits, masks)

        if train:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return loss.item(), logits

    @torch.no_grad()
    def _save_preview(self, imgs, logits, masks, step_tag: str):
        pred = logits.argmax(1).float().unsqueeze(1)   # (B,1,H,W)
        grid_dir = self.out_dir / "previews"
        grid_dir.mkdir(exist_ok=True, parents=True)
        save_image((imgs[:4].clamp(0,1)), grid_dir / f"{step_tag}_img.png")
        save_image((pred[:4] / (self.cfg["num_classes"] - 1)).clamp(0,1), grid_dir / f"{step_tag}_pred.png")

    def train(self):
        best_val = 1e9
        for epoch in range(1, self.epochs + 1):
            self.backbone.train(False)  # backbone stays frozen
            self.head.train(True)

            running = 0.0
            for batch in self.train_loader:
                loss, _ = self._step(batch, train=True)
                running += loss
            avg_train = running / max(1, len(self.train_loader))

            # val
            self.head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    loss, logits = self._step(batch, train=False)
                    val_loss += loss
                    if i == 0:
                        imgs, masks = batch
                        self._save_preview(imgs, logits, masks, f"ep{epoch:03d}")
            val_loss /= max(1, len(self.val_loader))

            print(f"[epoch {epoch}/{self.epochs}] train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

            # save best / periodic
            if (epoch % int(self.cfg["save_every"]) == 0) or (val_loss < best_val):
                best_val = min(best_val, val_loss)
                ckpt = {
                    "head": self.head.state_dict(),
                    "backbone_lora": {k: v for k, v in self.backbone.state_dict().items() if "lora_" in k},
                    "cfg": self.cfg,
                    "epoch": epoch,
                }
                torch.save(ckpt, self.out_dir / f"checkpoint_ep{epoch:03d}.pt")
