"""
Mixture-of-Depths (MoD) multi-modal transformer for DeepSense 6G beam prediction,
together with the training / evaluation loop.

The model fuses four modalities per time step (RGB image, LiDAR bird's-eye-view
histogram, radar range-Doppler tokens, GPS position) into a single token stream.
Each MoD block routes only a learnable fraction of the tokens through attention +
FFN, so compute scales with the learned keep-ratio instead of the full sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
def make_backbone(in_ch: int, d: int = 64):
    """ResNet-18 up to layer3 (32x32 output), first conv patched to `in_ch`
    channels, and the feature map projected down to `d` channels."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_ch != 3:
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    body = nn.Sequential(*list(m.children())[:6])       # conv1 ... layer3
    head_conv = nn.Conv2d(128, d, kernel_size=1)         # reduce channels to d
    return nn.Sequential(body, head_conv)


class MoDBlock(nn.Module):
    """A single Mixture-of-Depths transformer block.

    A small router scores every token; only the top-`k` tokens (where `k` is set
    by the learnable, differentiable `ratio`) pass through attention + FFN, and
    the result is scattered back into the full sequence. During training the
    ratio is interpolated between its two nearest bins; at eval it snaps to the
    nearest bin for a single, cheap forward pass.
    """
    def __init__(self, d, n_head=8, ffn_mult=4, r_init=1.0,
                 bins=torch.linspace(0.1, 1.0, 10)):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)
        )
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d)
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        self.ratio = nn.Parameter(torch.tensor(r_init))   # learnable keep-ratio
        self.register_buffer("bins", bins)                # candidate ratios 0.1..1.0

    def _single_pass(self, x, k):
        """Attention + FFN over only the top-`k` routed tokens."""
        B, L, d = x.shape
        scores = self.router(x).squeeze(-1)                       # (B, L)
        scores_topk, idx = torch.topk(scores, k, dim=1, sorted=False)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
        x_sel = torch.gather(x, 1, idx_exp)

        y = self.attn(self.norm1(x_sel), self.norm1(x_sel),
                      self.norm1(x_sel), need_weights=False)[0]
        y = x_sel + y
        y = y + self.ffn(self.norm2(y))

        # gate the selected tokens by their (sigmoid) router score
        y = y * torch.sigmoid(scores_topk).unsqueeze(-1)          # (B, k, 1)
        out = x.clone()
        return out.scatter_add(1, idx_exp, y)

    def _two_pass(self, x, k_low, k_high, w_high):
        """Blend the low-k and high-k passes so `ratio` stays differentiable."""
        low_out = self._single_pass(x, k_low)
        high_out = self._single_pass(x, k_high)
        return (1 - w_high) * low_out + w_high * high_out

    def forward(self, x):
        if not self.training:                     # eval -> single, snapped path
            r = torch.clamp(self.ratio, 0.1, 1.0).detach()
            r = self.bins[torch.argmin(torch.abs(self.bins - r))]
            k = max(1, int(torch.ceil(r * x.size(1)).item()))
            return self._single_pass(x, k)

        # training -> interpolate between the two nearest bins
        r = torch.clamp(self.ratio, 0.1, 1.0)
        diffs = torch.abs(self.bins - r)
        hi, lo = torch.topk(diffs, 2, largest=False).indices
        r_lo, r_hi = self.bins[lo], self.bins[hi]
        w_hi = (r - r_lo) / (r_hi - r_lo)
        k_lo = torch.ceil(r_lo * x.size(1)).int()
        k_hi = torch.ceil(r_hi * x.size(1)).int()
        return self._two_pass(x, k_lo.item(), k_hi.item(), w_hi)


class MoD_model(nn.Module):
    """Multi-modal (RGB + LiDAR-BEV + radar + GPS) Mixture-of-Depths transformer."""
    def __init__(self, n_beams, n_frames=5, d=64, n_layers=8, k_bins=10,
                 r_init=1.0, use_radar=True, use_gps=True):
        super().__init__()
        self.rgb_backbone = make_backbone(3, d)
        self.bev_backbone = make_backbone(1, d)

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.time_emb = nn.Embedding(n_frames, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, 32 * 32, d))    # 32x32 visual tokens
        self.mod_emb = nn.Embedding(4, d)                          # rgb=0, bev=1, radar=2, gps=3

        self.use_radar = use_radar
        self.use_gps = use_gps
        if use_radar:
            self.radar_proj = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        if use_gps:
            self.gps_proj = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))

        self.encoder = nn.Sequential(*[
            MoDBlock(d, r_init=r_init, n_head=8, ffn_mult=4,
                     bins=torch.linspace(0.1, 1.0, k_bins))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d, n_beams)

    def _to_tokens(self, feat, B, T):
        # (B*T, d, 32, 32) -> (B*T, N, d) -> (B, T, N, d)
        feat = feat.flatten(2).transpose(1, 2)
        return feat.view(B, T, -1, feat.size(-1))

    def forward(self, rgb, bev, radar=None, gps=None):
        B, T, _, H, W = rgb.shape
        rgb_tok = self._to_tokens(self.rgb_backbone(rgb.view(-1, 3, H, W)), B, T)
        bev_tok = self._to_tokens(self.bev_backbone(bev.view(-1, 1, H, W)), B, T)
        N = rgb_tok.size(2)
        pos = self.pos_emb[:, :N, :]

        seq = []
        for t in range(T):
            te = self.time_emb(torch.tensor(t, device=rgb.device))
            seq.append(rgb_tok[:, t] + pos + self.mod_emb.weight[0] + te)
            seq.append(bev_tok[:, t] + pos + self.mod_emb.weight[1] + te)
            if self.use_radar and radar is not None:
                seq.append(self.radar_proj(radar[:, t]) + self.mod_emb.weight[2] + te)
            if self.use_gps and gps is not None:
                seq.append(self.gps_proj(gps[:, t]) + self.mod_emb.weight[3] + te)

        seq = torch.cat(seq, dim=1)                    # (B, L, d)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)             # prepend CLS
        out = self.encoder(seq)
        return self.head(out[:, 0])                    # classify from CLS token


# --------------------------------------------------------------------------- #
#  Training utilities
# --------------------------------------------------------------------------- #
def mod_ratio_loss(model, target):
    """MSE between the average learned keep-ratio and the target ratio."""
    ratios = [b.ratio for b in model.encoder]
    avg_r = torch.mean(torch.stack(ratios))
    target = torch.tensor(target, device=avg_r.device)
    return F.mse_loss(avg_r, target), avg_r.item()


@torch.no_grad()
def topk_accuracy(logits, target, topk=(1, 3, 5)):
    """Top-k accuracy for each k in `topk` over the current mini-batch."""
    max_k = max(topk)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.unsqueeze(1))
    return [correct[:, :k].reshape(-1).float().sum() / target.size(0) for k in topk]


def run_epoch(model, loader, epoch, warmup_epochs, ratio_target, ratio_lambda,
              optim=None, scaler=None, device="cuda"):
    """Run one train (optim given) or eval (optim None) epoch. Returns
    (loss, top1, top3, top5, avg_ratio) averaged over the epoch."""
    is_train = optim is not None
    model.train() if is_train else model.eval()

    cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss, n = 0.0, 0
    total_acc = {1: 0.0, 3: 0.0, 5: 0.0}
    tot_ratio, avg_ratio = 0.0, 1.0

    for (rgb, lidar, radar, gps), labels in loader:
        rgb, lidar = rgb.to(device), lidar.to(device)
        radar, gps = radar.to(device), gps.to(device)
        y = labels.to(device) - 1                 # DeepSense 6G beams are 1-indexed

        with torch.set_grad_enabled(is_train), \
             torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(rgb, lidar, radar, gps)
            loss = cross_entropy(logits, y)
            if epoch > warmup_epochs:
                loss_mod, avg_ratio = mod_ratio_loss(model, ratio_target)
                loss = loss + ratio_lambda * loss_mod

        if is_train:
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        acc1, acc3, acc5 = topk_accuracy(logits, y, (1, 3, 5))
        total_acc[1] += acc1.item() * bs
        total_acc[3] += acc3.item() * bs
        total_acc[5] += acc5.item() * bs
        tot_ratio += avg_ratio * bs
        n += bs

    return (total_loss / n, total_acc[1] / n, total_acc[3] / n,
            total_acc[5] / n, tot_ratio / n)
