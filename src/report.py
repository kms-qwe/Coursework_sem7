"""Сводная таблица prune_results.csv + benchmark.csv -> summary.md"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import out_dirs


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(v, digits=4):
    return "—" if v in (None, "") else f"{float(v):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _, results_dir = out_dirs(args.smoke)

    quality = read_csv(results_dir / "prune_results.csv")
    bench_path = results_dir / "benchmark.csv"
    bench = {r["model"]: r for r in read_csv(bench_path)} if bench_path.exists() else {}
    pruned = {(r["method"], r["ratio"]): r for r in quality if r["stage"] == "pruned"}
    final = [r for r in quality if r["stage"] in ("base", "finetuned")]
    base_b = bench.get("base", {})

    def speedup(b, key):
        if not b.get(key) or not base_b.get(key):
            return "—"
        return f"{float(base_b[key]) / float(b[key]):.2f}×"

    lines = [
        "| Метод | Прунинг | Параметры, M | GMACs | Файл, МБ | gzip, МБ | Dice без дообуч. | Dice | IoU | PH2 Dice "
        "| CPU 1 поток, мс | CPU, мс | MPS, мс | ORT 1 поток, мс | Ускорение CPU 1п. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in final:
        name = Path(r["model_file"]).stem if r.get("model_file") else "base"
        b = bench.get(name, {})
        p = pruned.get((r["method"], r["ratio"]), {})
        lines.append(
            f"| {'исходная' if r['stage'] == 'base' else r['method']} | {float(r['ratio']) * 100:.0f}% "
            f"| {int(r['nonzero_params']) / 1e6:.2f} | {float(r['gmacs']):.2f} | {fmt(b.get('file_mb'), 1)} | {fmt(b.get('gzip_mb'), 1)} "
            f"| {fmt(p.get('test_dice'))} | {fmt(r['test_dice'])} | {fmt(r['test_iou'])} | {fmt(r['ph2_dice'])} "
            f"| {fmt(b.get('cpu1_ms'), 1)} | {fmt(b.get('cpu_ms'), 1)} | {fmt(b.get('mps_ms'), 1)} "
            f"| {fmt(b.get('ort1_ms'), 1)} | {speedup(b, 'cpu1_ms')} |")

    text = ("test ISIC 2016 (379) и PH2 (200), 256x256. Dice после дообучения. "
            "Для unstructured_l1 параметры ненулевые. Время: медиана, batch 1.\n\n" + "\n".join(lines) + "\n")
    out = results_dir / "summary.md"
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
