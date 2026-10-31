"""Бейзлайн прунинга: методы torch-pruning + неструктурный L1, потом дообучение.

Результаты в runs/results/prune_results.csv, модели в runs/models/. Посчитанное при перезапуске пропускается.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch.nn.utils.prune as nnprune

from common import (IMG_SIZE, evaluate, get_device, load_data, load_model, loss_fn, model_from_state_dict,
                    model_stats, out_dirs, save_model, seed_everything, smp, torch, tp, train)

ALL_METHODS = ["random", "l1", "fpgm", "bn_scale", "taylor", "unstructured_l1"]
IMPORTANCES = {
    "random": lambda: tp.importance.RandomImportance(),
    "l1": lambda: tp.importance.GroupMagnitudeImportance(p=1),
    "fpgm": lambda: tp.importance.FPGMImportance(p=2),
    "bn_scale": lambda: tp.importance.BNScaleImportance(),
    "taylor": lambda: tp.importance.GroupTaylorImportance(),
}
FIELDS = ["method", "ratio", "stage", "params", "nonzero_params", "gmacs", "params_frac", "gmacs_frac",
          "val_dice", "test_dice", "test_iou", "test_jac65", "test_dice_global", "ph2_dice", "ph2_iou",
          "prune_sec", "ft_min", "model_file"]


def prune_structured(model, method, ratio, train_data, batch_size, taylor_batches):
    device = next(model.parameters()).device
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    pruner = tp.pruner.MetaPruner(model, example, importance=IMPORTANCES[method](), pruning_ratio=ratio,
                                  ignored_layers=[model.segmentation_head])
    if method == "taylor":  # нужны градиенты
        model.eval()
        model.zero_grad()
        for k, (x, y) in enumerate(train_data.batches(batch_size, shuffle=True)):
            if k >= taylor_batches:
                break
            loss_fn(model(x), y).backward()
    pruner.step()
    model.zero_grad(set_to_none=True)


def prune_unstructured(model, ratio):
    convs = [(m, "weight") for m in model.modules()
             if isinstance(m, torch.nn.Conv2d) and m is not model.segmentation_head[0]]
    nnprune.global_unstructured(convs, pruning_method=nnprune.L1Unstructured, amount=ratio)


def without_masks(model):
    # вшиваем маски prune в веса
    sd = model.state_dict()
    plain = {}
    for k, v in sd.items():
        if k.endswith(".weight_orig"):
            plain[k[:-len("_orig")]] = v * sd[k[:-len("_orig")] + "_mask"]
        elif not k.endswith(".weight_mask"):
            plain[k] = v
    return model_from_state_dict(plain, next(model.parameters()).device)


def measure(model, data, method, ratio, stage, base=None, **extra):
    plain = without_masks(model)
    stats = model_stats(plain)
    ev = {name: evaluate(model, data[name]) for name in ["val", "test", "ph2"]}
    row = {"method": method, "ratio": ratio, "stage": stage, **stats,
           "params_frac": stats["nonzero_params"] / base["nonzero_params"] if base else 1.0,
           "gmacs_frac": stats["gmacs"] / base["gmacs"] if base else 1.0,
           "val_dice": ev["val"]["dice"], "test_dice": ev["test"]["dice"], "test_iou": ev["test"]["iou"],
           "test_jac65": ev["test"]["jac65"], "test_dice_global": ev["test"]["dice_global"],
           "ph2_dice": ev["ph2"]["dice"], "ph2_iou": ev["ph2"]["iou"], **extra}
    return row, plain


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  {row['method']:16s} {row['ratio']:.1f} {row['stage']:9s} параметры {row['params_frac']:.3f} "
          f"GMACs {row['gmacs_frac']:.3f}  test Dice {row['test_dice']:.4f}  PH2 Dice {row['ph2_dice']:.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taylor-batches", type=int, default=16, help="батчей для градиентов taylor")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон на 16 картинках")
    args = ap.parse_args()
    if args.smoke:
        args.ft_epochs, args.batch_size, args.taylor_batches = 1, 4, 2

    seed_everything(42)
    device = get_device(args.device)
    models_dir, results_dir = out_dirs(args.smoke)
    print(f"device {device} | torch {torch.__version__} | smp {smp.__version__} | torch-pruning {tp.__version__}")
    base_path = models_dir / "base.pth"
    if not base_path.exists():
        sys.exit(f"нет {base_path}, сначала train_base.py")

    data = load_data(device, args.smoke)
    results_csv = results_dir / "prune_results.csv"
    log_path = results_dir / "prune_ft_log.jsonl"

    done = set()
    if results_csv.exists():
        with open(results_csv) as f:
            done = {(r["method"], float(r["ratio"])) for r in csv.DictReader(f) if r["stage"] == "finetuned"}
    base_row, _ = measure(load_model(base_path, device), data, "none", 0.0, "base", model_file="models/base.pth")
    base = {"nonzero_params": base_row["nonzero_params"], "gmacs": base_row["gmacs"]}
    if not results_csv.exists():
        append_row(results_csv, base_row)

    for method in args.methods:
        for ratio in args.ratios:
            if (method, ratio) in done:
                print(f"=== {method} {ratio}: уже есть")
                continue
            print(f"=== {method}, прунинг {ratio:.0%}", flush=True)
            seed_everything(0)
            model = load_model(base_path, device)
            t0 = time.time()
            if method == "unstructured_l1":
                prune_unstructured(model, ratio)
            else:
                prune_structured(model, method, ratio, data["train"], args.batch_size, args.taylor_batches)
            prune_sec = round(time.time() - t0, 1)
            row, _ = measure(model, data, method, ratio, "pruned", base, prune_sec=prune_sec)
            append_row(results_csv, row)

            t0 = time.time()
            train(model, data["train"], data["val"], args.ft_epochs, args.ft_lr, args.batch_size, warmup_epochs=0,
                  log_path=log_path, tag=f"{method}_{int(ratio * 100)}")
            ft_min = round((time.time() - t0) / 60, 1)
            model_file = f"models/{method}_{int(ratio * 100)}.pth"
            row, plain = measure(model, data, method, ratio, "finetuned", base,
                                 prune_sec=prune_sec, ft_min=ft_min, model_file=model_file)
            save_model(plain, models_dir.parent / model_file, method=method, ratio=ratio)
            append_row(results_csv, row)
            del model, plain
            if device.type == "mps":
                torch.mps.empty_cache()
    print(f"-> {results_csv}")


if __name__ == "__main__":
    main()
