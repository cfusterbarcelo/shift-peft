# DINO-EM-PEFT
Parameter-efficient fine-tuning (PEFT) of DINOv2 ViTs for electron microscopy (EM) **foreground segmentation** using **LoRA** adapters.

The goal of this repo is to systematically study how DINO size and LoRA usage impact EM segmentation performance under domain shift.

---

> **Status**
>
> This repo is currently under active development.  
> It is structured to run:
>
> - **locally on a Mac with a GPU**, and  
> - **on an HPC cluster via `.sbatch` jobs**.
>
> For this reason, most experiment configs are duplicated so that the **same experiment** can be launched in both environments with minimal changes.

---

> **Automated reporting & visualization**
>
> - `scripts/summarize_seg_results.py` collects per-run metrics into `summary/*.csv`.
> - `scripts/plot_seg_summary.py` turns those CSVs into publication-ready plots and a small HTML dashboard.  
>   See [Visualizing results](#visualizing-results).


## TL;DR

```bash
# 0) create env & install package (Mac / local GPU example)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# (cluster) you will typically activate a conda/module env in your .sbatch script

# 1) prepare datasets (edit paths inside the script you use)
python scripts/<your_dataset_script>.py

# 2) launch an experiment by pointing to a config YAML
# local Mac GPU:
python scripts/<your_train_script>.py --config config/mac/<experiment>.yaml

# cluster (via sbatch):
sbatch scripts/slurm/<your_sbatch_script>.sbatch config/cluster/<experiment>.yaml

# 3) after several runs, summarize & visualize results
python scripts/summarize_seg_results.py
python scripts/plot_seg_summary.py
```
(Replace `<…>` placeholders with the actual script / config names you use in this repo.)

## Environments & Execution
This GitHub repo is intentionally set up to support two main environments:
1. **Local development on a Mac with GPU**  
   Use the YAMLs under `config/mac/` and the scripts under `scripts/` directly. Create a Python virtual environment and install the package in editable mode (`pip install -e .`).
2. **HPC cluster execution via SLURM**  
   Use the YAMLs under `config/cluster/` and the SLURM sbatch scripts under `scripts/slurm/`. Adjust the sbatch scripts to your cluster's module system and job scheduler settings. Create a conda environment or load modules as needed.

## Overview 
* Backbone: DINOv2 ViT (`vits14`/`vitb14` etc.) via `torch.hub`.
* PEFT: LoRA injected into attention `qkv` and `proj` linear layers
* Head: simple 1×1 conv projection → upsample to image size.
* Datasets: utility to compose Drosophila + Lucchi++ into a unified layout
* Devices: `device: "auto"` picks cuda → mps → cpu.
* Experiment tracking: MLflow integration for logging metrics, parameters, and artifacts.

## Results Layout & Experiment IDs

All training, evaluation, and feature/analysis scripts share a single results layout driven by the YAML config. Each config must define:

- `experiment_id`: follow `YYYY-MM-DD_SUBTASK_DATASETS_BACKBONE_LORA_TASK` (e.g. `2025-11-20_A1_lucchi+droso_dinov2-base_lora-none_seg`). Update this string before every new run so directories and config copies stay unique.
- `results_root`: the absolute (or `~`-expanded) directory where you keep all run outputs, e.g. `/Users/cfuste/Documents/Results/DINO-LoRA`.
- `task_type`: `"seg"` for segmentation training/eval (`train_em_seg.py`, `eval_em_seg.py`) and `"feats"` for feature extraction/PCA/latent scripts (`extract_features.py`, `run_pca.py`, future FID/LR scripts).

When one of the scripts starts, it creates `<results_root>/<task_type>/<experiment_id>/` and saves:

```
seg/
  <RunID>/
    config_used.yaml      # exact config snapshot
    run_info.txt          # timestamp, git hash, device, img size…
    metrics.json          # sections: train, eval
    logs/
    ckpts/best_model.pt, last_model.pt
    figs/previews/, figs/eval_previews/

feats/
  <RunID>/
    config_used.yaml
    run_info.txt
    metrics.json          # sections: features, pca, etc.
    features.npz          # feature extractor output
    plots/*.png           # PCA / UMAP or other analysis figures
```

`metrics.json` is automatically updated per phase: training adds best-val stats, `eval_em_seg.py` appends Lucchi/Droso IoU/Dice summaries, `extract_features.py` records feature dimensionality, and `run_pca.py` records PCA/UMAP configuration. All scripts keep writing MLflow artifacts as before.

To launch a new experiment:

1. Pick/clone the most relevant YAML under `config/`.
2. Edit the dataset paths as usual, then set a fresh `experiment_id`, ensure `results_root` points to your preferred root folder, and set `task_type` (`"seg"` or `"feats"`).
3. Run the desired script. Outputs, checkpoints, configs, and plots will land under the corresponding run directory so you can diff or archive them safely.

## Repository structure
```DINO-EM-PEFT/
  config/
    mac/                   # Local (Mac) experiment configs
    cluster/               # Cluster/SLURM experiment configs
    *.yaml                 # Legacy / shared configs pending cleanup
  docs/
    media/                 # Screenshots, plots, README assets
  scripts/                 # Training, evaluation, feature extraction, analysis
  slurm/                   # SLURM sbatch scripts for cluster execution
  src/                     # Package source code
    data/                  # Dataset loading, preprocessing, augmentation
    models/                # DINO + LoRA + segmentation head definitions
    training/              # Training loops, evaluation, metrics
    utils/                 # Helpers: logging, visualization, MLflow integration
  README.md
  pyproject.toml           # Package metadata
  .gitignore
```

## Datasets
Lucchi, A., Smith, K., Achanta, R., Knott, G., & Fua, P. (2011). Supervoxel-based segmentation of mitochondria in em image stacks with learned shape features. IEEE transactions on medical imaging, 31(2), 474-486. Download [here](https://casser.io/connectomics).

Casser, V., Kang, K., Pfister, H., & Haehn, D. (2020, September). Fast mitochondria detection for connectomics. In Medical Imaging with Deep Learning (pp. 111-120). PMLR. Download [here](https://github.com/unidesigner/groundtruth-drosophila-vnc/tree/master).

For usability purposes, the two dataset are composed into:
```bash
<BASE>/composed-dinopeft/
  train/images, train/masks
  test/images,  test/masks
  mapping.csv
```
With an 85% split for the Casser et al. dataset. 

To do so, download the original datasets and run:

```bash
python scripts/compose_em_datasets.py
```

## Acknowledgements
This project stands on the shoulders of excellent open-source work and research. We’re grateful to the authors and maintainers of the following projects and papers:

- **DINOv2 (Meta AI / Facebook Research)**  
  We use DINOv2 Vision Transformers and public pretraining weights (loaded via `torch.hub`) as our frozen backbone. DINOv2 provides strong, general-purpose visual representations that we adapt to electron microscopy via parameter-efficient fine-tuning (PEFT).  
  Repo: <https://github.com/facebookresearch/dinov2>

- **RobvanGastel/dinov3-finetune**  
  This repository informed practical design choices for LoRA-based finetuning of DINOv2/DINOv3 encoders: which linear layers to target (e.g., attention `qkv` and `proj`), how to organize training/evaluation code, and how to integrate PEFT cleanly around a frozen backbone.  
  Repo: <https://github.com/RobvanGastel/dinov3-finetune>

- **samar-khanna/ExPLoRA**  
  ExPLoRA provides a strong reference for adapting Vision Transformers under domain shift via extended pre-training. While our current baseline is supervised LoRA on DINOv2, ExPLoRA guides our roadmap toward semi/self-supervised adaptation on unlabeled EM volumes and domain-shifted pretraining strategies.  
  Repo: <https://github.com/samar-khanna/ExPLoRA>

- **DINOSim (Electron Microscopy zero-shot evaluation)**  
  DINOSim motivates our evaluation focus: it investigates zero-shot detection/segmentation on EM imagery using DINO features and highlights the domain gap for microscopy. We build on that insight by demonstrating how PEFT (LoRA) improves downstream EM segmentation compared to zero-shot.  
  Project/Paper: *DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Electron Microscopy Images.*

**Licensing note:**  
Please review and respect the licenses of upstream repositories (e.g. DINOv2) and any datasets you use. Their terms apply to model weights, code, and data redistributed or fine-tuned within this project.
