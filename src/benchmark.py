"""Замеры: параметры, GMACs, размер файла, время на batch 1 (torch cpu/mps, onnxruntime).

Режимы: cpu1, cpu, mps, ort1, ort. Пишется медиана и p90 в мс -> runs/results/benchmark.csv
"""
import argparse
import csv
import gzip
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnxruntime as ort

from common import IMG_SIZE, load_model, model_stats, out_dirs, sync, torch

ALL_MODES = ["cpu1", "cpu", "mps", "ort1", "ort"]


def time_fn(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def bench_torch(model, device, threads, warmup, runs):
    torch.set_num_threads(threads)
    model = model.to(device).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    def run():
        with torch.inference_mode():
            model(x)
        sync(device)

    return time_fn(run, warmup, runs)


def bench_ort(onnx_path, threads, warmup, runs):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)}
    return time_fn(lambda: sess.run(None, feed), warmup, runs)


def export_onnx(model, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model.cpu().eval(), (x,), str(path), input_names=["input"], output_names=["logits"],
                      dynamo=False, opset_version=18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--models", nargs="*", help="имена без .pth, по умолчанию все")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.runs, args.warmup = 5, 2

    models_dir, results_dir = out_dirs(args.smoke)
    onnx_dir = models_dir.parent / "onnx"
    all_threads = os.cpu_count() or 1
    paths = sorted(models_dir.glob("*.pth"))
    if args.models:
        paths = [p for p in paths if p.stem in args.models]
    if not paths:
        sys.exit(f"нет моделей в {models_dir}")
    print(f"{platform.platform()} | {platform.processor()} | потоков CPU: {all_threads} | "
          f"torch {torch.__version__} | onnxruntime {ort.__version__} | моделей: {len(paths)}")

    rows = []
    for path in paths:
        model = load_model(path, "cpu")
        stats = model_stats(model)
        raw = path.read_bytes()
        row = {"model": path.stem, **stats, "file_mb": len(raw) / 2**20,
               "gzip_mb": len(gzip.compress(raw, compresslevel=6)) / 2**20}
        onnx_path = onnx_dir / f"{path.stem}.onnx"
        if any(m.startswith("ort") for m in args.modes):
            export_onnx(model, onnx_path)
        for mode in args.modes:
            if mode == "mps" and not torch.backends.mps.is_available():
                continue
            if mode == "cpu1":
                med, p90 = bench_torch(model, torch.device("cpu"), 1, args.warmup, args.runs)
            elif mode == "cpu":
                med, p90 = bench_torch(model, torch.device("cpu"), all_threads, args.warmup, args.runs)
            elif mode == "mps":
                med, p90 = bench_torch(model, torch.device("mps"), all_threads, args.warmup, args.runs)
            elif mode == "ort1":
                med, p90 = bench_ort(onnx_path, 1, args.warmup, args.runs)
            else:
                med, p90 = bench_ort(onnx_path, all_threads, args.warmup, args.runs)
            row[f"{mode}_ms"], row[f"{mode}_p90_ms"] = med, p90
        model.cpu()
        rows.append(row)
        print(f"{path.stem:20s} {stats['nonzero_params'] / 1e6:6.2f}M  {stats['gmacs']:5.2f} GMACs  "
              + "  ".join(f"{m} {row[f'{m}_ms']:.1f} мс" for m in args.modes if f"{m}_ms" in row), flush=True)

    out = results_dir / "benchmark.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
