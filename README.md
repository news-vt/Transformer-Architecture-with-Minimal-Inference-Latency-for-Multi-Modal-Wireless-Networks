# Multi-Modal Beam Prediction with a Mixture-of-Depths Transformer

 The implementation of **mmWave beam prediction** on the
[DeepSense 6G](https://www.deepsense6g.net/) multi-modal dataset
(RGB camera + LiDAR + radar + GPS), using a **Mixture-of-Depths (MoD)** transformer.

Each sample is a sequence of **5 time steps**, and every step fuses four modalities
(RGB image, LiDAR bird's-eye-view histogram, radar range–Doppler tokens, GPS position)
into a single token stream. Each MoD encoder block learns a differentiable *keep-ratio*
and routes only the top-scoring fraction of tokens through attention + FFN, so compute
scales with the learned ratio instead of the full sequence. A CLS token is read out and
classified into one of the mmWave beam indices.

---

## Repository layout

```
.
├── main.py         # Entry point: data pipeline (DeepSense 6G) + training driver
├── training.py     # MoD model definition + training / evaluation loop
├── batchjob.sh     # Slurm submission script
└── README.md
```

> The **DeepSense 6G dataset is *not* included** in this repository. Download it
> separately and point the code at it via the `DEEPSENSE_ROOT` environment variable
> — see [Dataset setup](#dataset-setup).

---

## Requirements

- Linux + NVIDIA GPU with CUDA (training uses `torch.cuda.amp` mixed precision)
- Python 3.9+
- A Conda environment with these packages (versions this project was validated with):

  | Package | Version |
  |---------|---------|
  | torch | 2.1.2 (cu121) |
  | torchvision | 0.16.2 |
  | open3d | 0.19.0 |
  | numpy | 1.26.4 |
  | pandas | 2.2.3 |
  | pillow | 10.4.0 |
  | wandb | 0.17.4 |

### Create the environment

```bash
conda create -n pmod python=3.9 -y
conda activate pmod

# CUDA build of PyTorch (match your CUDA toolkit)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install open3d numpy pandas pillow wandb
```

---

## Dataset setup

This project uses the **DeepSense 6G multi-modal** data (the ML-challenge development
split). Download the multi-modal scenario(s) from <https://www.deepsense6g.net/> and
unpack them anywhere on disk.

### Expected layout

The loader reads one CSV that lists every sample and resolves the (relative) file paths
inside it:

```
<DEEPSENSE_ROOT>/                         # <-- set DEEPSENSE_ROOT to this folder
├── ml_challenge_dev_multi_modal.csv      # the manifest main.py reads
├── scenarioXX/unit1/camera_data/...      # RGB images  (unit1_rgb_1..5)
├── scenarioXX/unit1/lidar_data/...       # LiDAR .ply   (unit1_lidar_1..5)
├── scenarioXX/unit1/radar_data/...       # radar .npy   (unit1_radar_1..5)
├── scenarioXX/unit2/gps_data/...         # GPS .txt     (unit2_loc_1..2)
└── bev_cache/                            # auto-created LiDAR BEV cache
```

The CSV is expected to contain the columns
`unit1_rgb_{1..5}`, `unit1_lidar_{1..5}`, `unit1_radar_{1..5}`, `unit2_loc_{1..2}`,
and `unit1_beam` (64 beam classes; labels are 1-indexed and shifted to 0-indexed
internally).

### Point the code at your data (general path)

The dataset path is **not hard-coded** — it is read from the `DEEPSENSE_ROOT`
environment variable, which should point at the folder above (the one containing
`ml_challenge_dev_multi_modal.csv`):

```bash
export DEEPSENSE_ROOT=/path/to/your/deepsense/Multi_Modal
```

If it is unset, `main.py` falls back to `./data/Multi_Modal`.

> **First run is slow.** LiDAR point clouds are converted to 256×256 BEV histograms
> on the fly and cached under `<DEEPSENSE_ROOT>/bev_cache/`; later runs reuse the cache.

---

## Weights & Biases

Training logs to [W&B](https://wandb.ai/). Either log in once (`wandb login`) or disable
online logging:

```bash
export WANDB_MODE=offline      # or: export WANDB_DISABLED=true
```

---

## Running

### Locally

```bash
conda activate pmod
export DEEPSENSE_ROOT=/path/to/your/deepsense/Multi_Modal
python main.py --dataset d6g --seed 0 --n_layers 8
```

### On a Slurm cluster (`batchjob.sh`)

`batchjob.sh` is the cluster entry point. **Edit the `#SBATCH` directives, the conda
setup, and `DEEPSENSE_ROOT` for your environment**, then submit. The seed is taken from
the `$SEED` environment variable (defaults to `0`):

```bash
SEED=0 sbatch batchjob.sh
# or
sbatch --export=ALL,SEED=0 batchjob.sh
```

The script runs from the directory it was submitted from (`$SLURM_SUBMIT_DIR`), so
submit it from inside this repo.

---

## Command-line arguments (`main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset {d6g}` | `d6g` | Dataset selector (DeepSense 6G multi-modal). |
| `--seed` | `0` | Random seed (also sets deterministic cuDNN). |
| `--n_layers` | `8` | Number of MoD encoder blocks. |
| `--ratio_target` | `0.4` | Target average token keep-ratio for the MoD regularizer. |

Other hyperparameters (epochs = 30, batch size = 8, lr = 1e-4, weight decay = 1e-2,
`ratio_lambda` = 10, etc.) live in the `cfg` dict at the top of `main()` in `main.py`.

---

## Outputs

- **Best checkpoint** → `saved_models/mod_ratio_<r>_L<layers>_seed<s>.pt`
- **Per-epoch metrics CSV** → `saved_models/metrics_ratio_<r>_L<layers>_seed<s>.csv`
- **W&B run** with train/val loss, top-1/3/5 accuracy, and the average keep-ratio.

The console prints per-epoch train/val time, loss, top-k accuracy, and the ratio.