"""
Entry point: train the Mixture-of-Depths multi-modal transformer on the
DeepSense 6G beam-prediction task.

    python main.py --dataset d6g --seed 0 --n_layers 8

The DeepSense 6G data path is read from the DEEPSENSE_ROOT environment variable,
which should point at the `Multi_Modal` folder (the one that contains
`ml_challenge_dev_multi_modal.csv`). See README.md.
"""
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open3d as o3d
import wandb

from training import MoD_model, run_epoch


# --------------------------------------------------------------------------- #
#  Dataset location (kept general -- override with the DEEPSENSE_ROOT env var)
# --------------------------------------------------------------------------- #
DEEPSENSE_ROOT = os.environ.get("DEEPSENSE_ROOT", "./data/Multi_Modal")
CACHE_ROOT = os.path.join(DEEPSENSE_ROOT, "bev_cache")   # cached LiDAR BEV maps


# --------------------------------------------------------------------------- #
#  Preprocessing helpers
# --------------------------------------------------------------------------- #
def bev_path_from_ply(ply_path: str) -> str:
    """Map a LiDAR `.ply` path to its cached BEV `.npy` path (dir made on demand)."""
    rel = os.path.splitext(os.path.relpath(ply_path, start=DEEPSENSE_ROOT))[0] + ".npy"
    cache_path = os.path.join(CACHE_ROOT, rel)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    return cache_path


def filter_points_by_height(points):
    mask = (points[:, 2] >= points[:, 2].min()) & (points[:, 2] <= points[:, 2].max())
    return points[mask]


def lidar_to_histogram_features(lidar):
    """Convert a LiDAR point cloud into a 1x256x256 bird's-eye-view histogram."""
    x_min, x_max = np.percentile(lidar[:, 0], [5, 95])
    y_min, y_max = np.percentile(lidar[:, 1], [5, 95])
    pts = filter_points_by_height(lidar)

    hist_max_per_pixel, pad = 3, 2.0
    xbins = np.linspace(x_min - pad, x_max + pad, 257)
    ybins = np.linspace(y_min - pad, y_max + pad, 257)
    hist = np.histogramdd(pts[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    hist = hist / hist_max_per_pixel
    return hist[np.newaxis, :, :]


def radar_cube_to_tokens(
    cube,                       # complex np.ndarray (Rx, Ns, Nchirps), e.g. (4, 256, 250)
    fixed_points=300,           # number of tokens to output
    fc_hz=77e9,
    fs=6.2e6,
    slope=8.014e12,             # Hz/s
    tramp=47.5e-6,
    tidle=2e-6,
    rx_spacing_wavelengths=0.5,  # ~ lambda/2 ULA
    C0=299792458.0,
):
    """Turn a complex radar cube into `fixed_points` tokens of
    [velocity, azimuth, range, power_dB] via range/Doppler FFT + Bartlett AoA."""
    assert np.iscomplexobj(cube), f"Expected complex radar cube, got dtype={cube.dtype}"
    RX, Ns, Nc = cube.shape
    lam = C0 / fc_hz
    Tc = tramp + tidle
    prf = 1.0 / Tc

    # windowing in fast/slow time
    w_r = np.hanning(Ns).astype(np.float32)
    w_d = np.hanning(Nc).astype(np.float32)
    cube = cube * w_r[None, :, None] * w_d[None, None, :]

    # range FFT (axis=1), Doppler FFT (axis=2)
    R = np.fft.fft(cube, n=Ns, axis=1)
    D = np.fft.fft(R, n=Nc, axis=2)
    D = np.fft.fftshift(D, axes=2)

    # power map summed across Rx
    P = (np.abs(D) ** 2).sum(axis=0)                # (Nr, Nd)
    Nr, Nd = P.shape
    flat = P.ravel()

    # pick the top-K power bins deterministically
    if fixed_points <= flat.size:
        idx = np.argpartition(flat, -fixed_points)[-fixed_points:]
        idx = idx[np.argsort(flat[idx])][::-1]
    else:
        idx = np.argsort(flat)[::-1]
        reps = int(np.ceil(fixed_points / idx.size))
        idx = np.tile(idx, reps)[:fixed_points]

    r_bins = idx // Nd
    d_bins = idx % Nd

    # bins -> physical units
    fb = r_bins.astype(np.float64) * (fs / Ns)
    ranges = (C0 * fb) / (2.0 * slope)                          # meters
    fD = (d_bins.astype(np.float64) - Nd / 2.0) * (prf / Nd)
    vels = (lam * fD) / 2.0                                      # m/s

    # AoA via Bartlett beamforming over an angle grid
    d_m = rx_spacing_wavelengths * lam
    ang_grid = np.deg2rad(np.linspace(-80.0, 80.0, 161))
    pos = np.arange(RX, dtype=np.float64) * d_m
    SV = np.exp(-1j * (2.0 * np.pi / lam) * (pos[:, None] * np.sin(ang_grid)[None, :]))

    azis = []
    for rb, db in zip(r_bins, d_bins):
        v_rx = D[:, rb, db]
        resp = np.abs(SV.conj().T @ v_rx) ** 2
        azis.append(ang_grid[np.argmax(resp)])
    azis = np.array(azis, dtype=np.float32)

    p_sel = P[r_bins, d_bins]
    p_db = 10.0 * np.log10(p_sel + 1e-12).astype(np.float32)

    tokens = np.stack([vels.astype(np.float32), azis,
                       ranges.astype(np.float32), p_db], axis=1)   # (K, 4)
    return tokens


def data_split(split, n_samples):
    """Sequentially split `n_samples` into train/val indices by ratio `split`."""
    if not (-1e-9 < sum(split) - 1 < 1e-9):
        raise ValueError("Sum of data_split must be 1.")
    order = np.arange(n_samples)
    cut = int(split[0] * n_samples)
    return order[:cut], order[cut:]


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class FivePairsDataset(Dataset):
    """Each sample is a length-5 sequence of (RGB, LiDAR, radar, GPS) file paths;
    modalities are loaded/preprocessed on the fly (LiDAR BEV maps are cached)."""
    def __init__(self, X, y, transform_img=None):
        self.X = X
        self.y = y
        self.transform_img = transform_img

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_seq, pc_seq, radar_seq, gps_seq = [], [], [], []
        for (img_path, lidar_path, radar_path, gps_path) in self.X[idx]:
            # RGB
            img = Image.open(img_path).convert("RGB")
            img_seq.append(self.transform_img(img))

            # LiDAR -> cached BEV histogram
            cache_file = bev_path_from_ply(lidar_path)
            if os.path.exists(cache_file):
                pc = np.load(cache_file)
            else:
                pts = np.asarray(o3d.io.read_point_cloud(lidar_path).points)
                pc = lidar_to_histogram_features(pts)
                np.save(cache_file, pc)
            pc_seq.append(torch.from_numpy(pc).float())

            # radar -> tokens
            radar_seq.append(torch.from_numpy(radar_cube_to_tokens(np.load(radar_path))))

            # GPS (lat, lon, 0)
            gps = np.loadtxt(gps_path, dtype=np.float32).reshape(-1)
            gps_seq.append(torch.tensor([[float(gps[0]), float(gps[1]), 0]], dtype=torch.float32))

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return (torch.stack(img_seq), torch.stack(pc_seq),
                torch.stack(radar_seq), torch.stack(gps_seq)), label


# --------------------------------------------------------------------------- #
#  Training driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["d6g"], default="d6g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio_target", type=float, default=0.4,
                    help="target average token keep-ratio for the MoD regularizer")
    ap.add_argument("--n_layers", type=int, default=8)
    args = ap.parse_args()

    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)

    cfg = {
        "val_split": 0.1,
        "num_workers": 12,
        "batch_size": 8,
        "epochs": 30,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "ratio_target": args.ratio_target,
        "ratio_lambda": 10,
        "warmup_epochs": 0,
        "n_layers": args.n_layers,
        "n_beams": 64,                       # DeepSense 6G multi-modal: 64 beams
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "model_path": os.path.join(
            save_dir, f"mod_ratio_{args.ratio_target}_L{args.n_layers}_seed{args.seed}.pt"),
        "metrics_path": os.path.join(
            save_dir, f"metrics_ratio_{args.ratio_target}_L{args.n_layers}_seed{args.seed}.csv"),
    }
    print(cfg)

    # ---- reproducibility ----
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    wandb.init(
        project="deepsense6g_mod",
        name=f"seed{args.seed}_ratio{args.ratio_target}_L{args.n_layers}",
        config={"seed": args.seed, "ratio_target": args.ratio_target,
                "n_layers": args.n_layers},
    )

    # ---- transforms ----
    tr_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    te_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---- build the sample table from the challenge CSV ----
    csv_path = os.path.join(DEEPSENSE_ROOT, "ml_challenge_dev_multi_modal.csv")
    print(f"Loading DeepSense 6G split from: {csv_path}")
    df = pd.read_csv(csv_path)

    inputs, labels = [], []
    for i in range(len(df)):
        pair_seq = []
        for t in range(1, 6):                        # 5 time steps
            img_path = os.path.join(DEEPSENSE_ROOT, df[f"unit1_rgb_{t}"][i][2:])
            lidar_path = os.path.join(DEEPSENSE_ROOT, df[f"unit1_lidar_{t}"][i][2:])
            radar_path = os.path.join(DEEPSENSE_ROOT, df[f"unit1_radar_{t}"][i][2:])
            gps_col = f"unit2_loc_{t}" if t < 3 else "unit2_loc_2"
            gps_path = os.path.join(DEEPSENSE_ROOT, df[gps_col][i][2:])
            pair_seq.append((img_path, lidar_path, radar_path, gps_path))
        inputs.append(pair_seq)
        labels.append(df["unit1_beam"][i])

    X = np.array(inputs)
    y = np.array(labels)

    tr_idx, val_idx = data_split([1 - cfg["val_split"], cfg["val_split"]], len(X))
    tr_dataset = FivePairsDataset(X[tr_idx], y[tr_idx], transform_img=tr_transform)
    val_dataset = FivePairsDataset(X[val_idx], y[val_idx], transform_img=te_transform)

    tr_loader = DataLoader(tr_dataset, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # ---- model / optimizer ----
    device = cfg["device"]
    model = MoD_model(n_beams=cfg["n_beams"], n_layers=cfg["n_layers"],
                      use_radar=True, use_gps=True).to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["epochs"])

    # ---- train ----
    best_acc, records = 0.0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr1, tr3, tr5, tr_ratio = run_epoch(
            model, tr_loader, epoch, cfg["warmup_epochs"], cfg["ratio_target"],
            cfg["ratio_lambda"], optim, scaler, device)
        t1 = time.time()
        val_loss, v1, v3, v5, v_ratio = run_epoch(
            model, val_loader, epoch, cfg["warmup_epochs"], cfg["ratio_target"],
            cfg["ratio_lambda"], device=device)
        t2 = time.time()
        scheduler.step()

        print(f"Epoch {epoch:02d}/{cfg['epochs']} | "
              f"train {t1 - t0:4.1f}s loss {tr_loss:.3f} top1 {tr1:.3f} | "
              f"val {t2 - t1:4.1f}s loss {val_loss:.3f} "
              f"top1 {v1:.3f} top3 {v3:.3f} top5 {v5:.3f} | ratio {v_ratio:.3f}")

        if v1 > best_acc:
            best_acc = v1
            torch.save(model.state_dict(), cfg["model_path"])
            print("  ↳ new best model saved")

        row = {"epoch": epoch, "train_loss": tr_loss, "train_top1": tr1,
               "train_top3": tr3, "train_top5": tr5, "val_loss": val_loss,
               "val_top1": v1, "val_top3": v3, "val_top5": v5,
               "ratio": v_ratio, "seed": args.seed}
        wandb.log(row)
        records.append(row)

    wandb.finish()
    pd.DataFrame(records).to_csv(cfg["metrics_path"], index=False)
    print(f"Training done. Best val top-1: {best_acc:.3f}")


if __name__ == "__main__":
    main()
