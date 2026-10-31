"""Обучение исходной модели -> runs/models/base.pth"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (IMG_SIZE, build_model, evaluate, get_device, load_data, model_stats, out_dirs, save_model,
                    seed_everything, smp, torch, tp, train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps / cpu / cuda")
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон на 16 картинках")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.batch_size = 1, 4

    seed_everything(42)
    device = get_device(args.device)
    models_dir, results_dir = out_dirs(args.smoke)
    print(f"device {device} | torch {torch.__version__} | smp {smp.__version__} | torch-pruning {tp.__version__}")

    data = load_data(device, args.smoke)
    print(", ".join(f"{k}: {len(v)}" for k, v in data.items()))

    log_path = results_dir / "base_train_log.jsonl"
    log_path.unlink(missing_ok=True)
    model = build_model(encoder_weights="imagenet").to(device)
    t0 = time.time()
    history = train(model, data["train"], data["val"], args.epochs, args.lr, args.batch_size,
                    log_path=log_path, tag="base")
    train_min = (time.time() - t0) / 60

    metrics = {name: evaluate(model, data[name]) for name in ["val", "test", "ph2"]}
    stats = model_stats(model)
    model_path = models_dir / "base.pth"
    save_model(model, model_path, method="none", ratio=0.0)

    summary = {
        "model": "smp.Unet(resnet34), 1 класс, энкодер ImageNet", "img_size": IMG_SIZE,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "device": str(device),
        "best_epoch": max(history, key=lambda r: r["val_dice"])["epoch"], "train_min": round(train_min, 1),
        "stats": stats, "file_mb": round(model_path.stat().st_size / 2**20, 1), "metrics": metrics,
    }
    (results_dir / "base_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nПараметры {stats['params'] / 1e6:.2f}M | GMACs {stats['gmacs']:.2f} | файл {summary['file_mb']} МБ | "
          f"обучение {summary['train_min']} мин, лучшая эпоха {summary['best_epoch']}")
    print(f"{'':6s} {'Dice':>7s} {'IoU':>7s} {'Jac65':>7s}")
    for name, m in metrics.items():
        print(f"{name:6s} {m['dice']:7.4f} {m['iou']:7.4f} {m['jac65']:7.4f}")
    print(f"-> {model_path}")


if __name__ == "__main__":
    main()
