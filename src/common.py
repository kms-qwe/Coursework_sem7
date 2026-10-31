"""Общие функции: данные, модель, обучение, метрики."""
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch_pruning as tp

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"
IMG_SIZE = 256
SPLITS = ["train", "val", "test", "ph2"]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def out_dirs(smoke=False):
    base = ROOT / "runs_smoke" if smoke else RUNS_DIR
    models, results = base / "models", base / "results"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return models, results


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name, limit=None):
    path = CACHE_DIR / f"{name}_{IMG_SIZE}.npz"
    if not path.exists():
        raise FileNotFoundError(f"нет {path}, запусти prepare_data.py")
    d = np.load(path)
    X, Y = d["X"], d["Y"]
    return (X[:limit], Y[:limit]) if limit else (X, Y)


class GpuData:
    # весь сплит держим на gpu, аугментации тоже там

    def __init__(self, X, Y, device):
        self.x = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().to(device)
        self.y = torch.from_numpy(Y).unsqueeze(1).to(device)
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def batches(self, batch_size, shuffle=False, augment=False, drop_last=False):
        n = len(self)
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        stop = n - n % batch_size if drop_last and n >= batch_size else n
        for i in range(0, stop, batch_size):
            j = idx[i:i + batch_size]
            x = self.x[j].float() / 255.0
            y = self.y[j].float()
            if augment:
                x, y = augment_batch(x, y)
            yield normalize(x), y


def load_data(device, smoke=False):
    limit = 16 if smoke else None
    return {name: GpuData(*load_split(name, limit), device) for name in SPLITS}


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment_batch(x, y):
    # флипы, поворот, масштаб 0.8-1.2, сдвиг, цвет +-20%
    b, dev = x.shape[0], x.device

    def rnd(*shape):
        return torch.rand(*shape, device=dev)

    flip = (rnd(b) < 0.5).view(b, 1, 1, 1)
    x, y = torch.where(flip, x.flip(3), x), torch.where(flip, y.flip(3), y)
    flip = (rnd(b) < 0.5).view(b, 1, 1, 1)
    x, y = torch.where(flip, x.flip(2), x), torch.where(flip, y.flip(2), y)

    angle = (rnd(b) * 2 - 1) * math.pi
    scale = 1.0 / (0.8 + 0.4 * rnd(b))
    shift = (rnd(b, 2) * 2 - 1) * 0.1
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = (F.grid_sample(y, grid, mode="bilinear", padding_mode="reflection", align_corners=False) > 0.5).float()

    brightness = 1 + (rnd(b, 1, 1, 1) * 2 - 1) * 0.2
    contrast = 1 + (rnd(b, 1, 1, 1) * 2 - 1) * 0.2
    saturation = 1 + (rnd(b, 1, 1, 1) * 2 - 1) * 0.2
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = x.mean(1, keepdim=True)
    x = ((x - gray) * saturation + gray) * brightness
    return x.clamp(0, 1), y


def build_model(encoder_weights="imagenet"):
    return smp.Unet("resnet34", encoder_weights=encoder_weights, classes=1, activation=None)


def save_model(model, path, **meta):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    return model_from_state_dict(ckpt["state_dict"], device)


def model_from_state_dict(sd, device="cpu"):
    # у прореженной модели меняем число каналов под размеры тензоров из sd
    model = build_model(encoder_weights=None)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            w = sd[f"{name}.weight"]
            module.out_channels, module.in_channels = w.shape[0], w.shape[1] * module.groups
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.num_features = sd[f"{name}.weight"].shape[0]
    for key, value in sd.items():
        owner_name, attr = key.rsplit(".", 1)
        owner = model.get_submodule(owner_name)
        if attr in owner._parameters:
            owner._parameters[attr] = torch.nn.Parameter(value.detach().clone())
        else:
            owner._buffers[attr] = value.detach().clone()
    return model.to(device).eval()


def loss_fn(logits, y):
    # bce + soft dice
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(model, data, batch_size=32):
    # dice и iou по картинкам (порог 0.5), jac65 как в ISIC 2018
    model.eval()
    dices, ious, inter_sum, den_sum = [], [], 0.0, 0.0
    for x, y in data.batches(batch_size):
        p, t = model(x) > 0, y > 0.5
        inter = (p & t).sum((1, 2, 3)).float()
        ps, ts = p.sum((1, 2, 3)).float(), t.sum((1, 2, 3)).float()
        union = ps + ts - inter
        ones = torch.ones_like(inter)
        dices.append(torch.where(ps + ts > 0, 2 * inter / (ps + ts).clamp(min=1), ones))
        ious.append(torch.where(union > 0, inter / union.clamp(min=1), ones))
        inter_sum += inter.sum().item()
        den_sum += (ps + ts).sum().item()
    dice, iou = torch.cat(dices), torch.cat(ious)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "jac65": torch.where(iou >= 0.65, iou, torch.zeros_like(iou)).mean().item(),
        "dice_global": 2 * inter_sum / max(den_sum, 1),
    }


def train(model, train_data, val_data, epochs, lr, batch_size=16, weight_decay=1e-4, warmup_epochs=1,
          log_path=None, tag=""):
    # adamw + cosine, в конце берем лучшую эпоху по val dice
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(train_data) // batch_size, 1)
    total, warmup = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_dice, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_data.batches(batch_size, shuffle=True, augment=True, drop_last=True):
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]
        val = evaluate(model, val_data)
        row = {"tag": tag, "epoch": epoch, "loss": loss_sum / max(n, 1), "val_dice": val["dice"],
               "val_iou": val["iou"], "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        history.append(row)
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] эпоха {epoch}/{epochs}  loss {row['loss']:.4f}  val Dice {val['dice']:.4f}  "
              f"IoU {val['iou']:.4f}  {row['sec']:.0f} с", flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    model.load_state_dict(best_state)
    return history


def model_stats(model):
    model.eval()
    device = next(model.parameters()).device
    macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return {"params": int(params), "nonzero_params": nonzero, "gmacs": macs / 1e9}
