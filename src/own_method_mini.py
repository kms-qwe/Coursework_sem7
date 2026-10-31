"""Свой метод, черновик на одном уровне (по умолчанию 70%).

Доли каналов по группам ищутся вокруг подсказки из unstructured_l1_90, кандидаты отсеиваются по
GMACs/весам, быстро оцениваются после пересчета BN, лучшие дообучаются. Для победителя потом GReg.
"""
import argparse
import copy
import csv
import gzip
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from benchmark import ALL_MODES, bench_ort, bench_torch, export_onnx
from common import (IMG_SIZE, evaluate, get_device, load_data, load_model, loss_fn, model_stats, out_dirs,
                    save_model, seed_everything, torch, tp, train)
from prune_baseline import append_row, measure

ROOT_TYPES = [torch.nn.Conv2d]


def head_conv(model):
    return model.segmentation_head[0]


def prunable_groups(model, example):
    DG = tp.DependencyGraph().build_dependency(model, example)
    names = {m: n for n, m in model.named_modules()}
    out = []
    for g in DG.get_all_groups(ignored_layers=[head_conv(model)], root_module_types=ROOT_TYPES):
        root = g[0].dep.target.module
        if root is head_conv(model) or DG.get_out_channels(root) <= 1:
            continue
        out.append((names[root], g))
    return DG, out


def apply_plan(model, example, remove):
    # remove: {имя свертки: [каналы]}
    DG = tp.DependencyGraph().build_dependency(model, example)
    names = {m: n for n, m in model.named_modules()}
    for g in DG.get_all_groups(ignored_layers=[head_conv(model)], root_module_types=ROOT_TYPES):
        idxs = remove.get(names.get(g[0].dep.target.module))
        if idxs:
            g.prune(idxs=list(idxs))
    return model


def count_channels(model, example):
    _, groups = prunable_groups(model, example)
    return sum(g[0].dep.target.module.out_channels for _, g in groups)


def prior_keep(base_cpu, unstructured_path, example):
    # доля ненулевых весов в каждой группе у неструктурной модели
    sd = torch.load(unstructured_path, map_location="cpu")["state_dict"]
    _, groups = prunable_groups(base_cpu, example)
    names = {m: n for n, m in base_cpu.named_modules()}
    q = {}
    for root_name, g in groups:
        nz, tot = 0, 0
        for dep, _ in g:
            layer = dep.layer
            if isinstance(layer, torch.nn.Conv2d) and dep.pruning_fn == tp.pruner.function.prune_conv_out_channels \
                    and layer is not head_conv(base_cpu):
                w = sd[f"{names[layer]}.weight"]
                nz += int((w != 0).sum())
                tot += w.numel()
        q[root_name] = nz / max(tot, 1)
    return q


def channels_for(shape, C, floor, target_keep):
    # keep ~ k*shape*C, k ищем бинпоиском под target_keep
    lo, hi = 0.0, 1e3
    for _ in range(80):
        k = (lo + hi) / 2
        keep = np.clip(np.round(k * shape * C), np.minimum(C, floor), C).astype(int)
        if keep.sum() < target_keep:
            lo = k
        else:
            hi = k
    return np.clip(np.round(hi * shape * C), np.minimum(C, floor), C).astype(int)


def cost_of(base_cpu, example, names, C, keep):
    m = copy.deepcopy(base_cpu)
    apply_plan(m, example, {n: list(range(k, c)) for n, c, k in zip(names, C, keep) if k < c})
    s = model_stats(m)
    return s["gmacs"], s["params"]


def l1_order(base, example):
    imp = tp.importance.GroupMagnitudeImportance(p=1)
    _, groups = prunable_groups(base, example)
    return {n: torch.argsort(imp(g).cpu()).tolist() for n, g in groups}


def remove_plan(order, names, C, keep):
    return {n: sorted(order[n][:c - k]) for n, c, k in zip(names, C, keep) if k < c}


@torch.no_grad()
def recalibrate_bn(model, data, n_images=None, batch_size=16):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    seen = 0
    for x, _ in data.batches(batch_size):
        model(x)
        seen += x.shape[0]
        if n_images and seen >= n_images:
            break
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.1
    model.eval()


def greg_targets(model, example, remove):
    _, groups = prunable_groups(model, example)
    targets = []
    for root_name, g in groups:
        rm = set(remove.get(root_name, []))
        if not rm:
            continue
        for item in g:
            dep, layer = item.dep, item.dep.layer
            pos = [i for i, r in zip(item.idxs, item.root_idxs) if r in rm]
            if not pos:
                continue
            idx = torch.tensor(pos, device=next(model.parameters()).device)
            if isinstance(layer, torch.nn.BatchNorm2d):
                targets += [(layer.weight, idx), (layer.bias, idx)]
            elif isinstance(layer, torch.nn.Conv2d) and dep.pruning_fn == tp.pruner.function.prune_conv_out_channels:
                targets.append((layer.weight, idx))
                if layer.bias is not None:
                    targets.append((layer.bias, idx))
    return targets


def greg_phase(model, data, targets, epochs, lr, lam_max, ramp=0.7, batch_size=16, log_path=None, tag=""):
    # после каждого шага удаляемые каналы *= (1 - lam), lam линейно растет до lam_max
    # (через L2 в лоссе с adam почти не работает)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * max(len(data["train"]) // batch_size, 1)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in data["train"].batches(batch_size, shuffle=True, augment=True, drop_last=True):
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            lam = lam_max * min(step / max(ramp * total, 1), 1.0)
            with torch.no_grad():
                for p, idx in targets:
                    p[idx] *= (1 - lam)
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]
        with torch.no_grad():
            norm = sum(float(p[idx].abs().sum()) for p, idx in targets)
        val = evaluate(model, data["val"])
        print(f"[{tag}] GReg эпоха {epoch}/{epochs}  loss {loss_sum / max(n, 1):.4f}  val Dice {val['dice']:.4f}  "
              f"lam {lam:.4f}  L1 гасимых {norm:.4g}  {time.time() - t0:.0f} с", flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(f'{{"tag": "{tag}", "greg_epoch": {epoch}, "loss": {loss_sum / max(n, 1)}, '
                        f'"val_dice": {val["dice"]}, "lam": {lam}, "l1_targets": {norm}}}\n')


def benchmark_models(paths, onnx_dir, out_csv, runs=100, warmup=20):
    all_threads = os.cpu_count() or 1
    old = {}
    if out_csv.exists():
        with open(out_csv) as f:
            old = {r["model"]: r for r in csv.DictReader(f)}
    for path in paths:
        model = load_model(path, "cpu")
        stats = model_stats(model)
        raw = path.read_bytes()
        row = {"model": path.stem, **stats, "file_mb": len(raw) / 2**20,
               "gzip_mb": len(gzip.compress(raw, compresslevel=6)) / 2**20}
        onnx_path = onnx_dir / f"{path.stem}.onnx"
        export_onnx(model, onnx_path)
        for mode in ALL_MODES:
            if mode == "mps" and not torch.backends.mps.is_available():
                continue
            if mode in ("cpu1", "cpu", "mps"):
                dev = torch.device("mps" if mode == "mps" else "cpu")
                med, p90 = bench_torch(model, dev, 1 if mode == "cpu1" else all_threads, warmup, runs)
            else:
                med, p90 = bench_ort(onnx_path, 1 if mode == "ort1" else all_threads, warmup, runs)
            row[f"{mode}_ms"], row[f"{mode}_p90_ms"] = med, p90
        model.cpu()
        old[path.stem] = row
        print(f"  {path.stem:20s} {stats['params'] / 1e6:6.2f}M  {stats['gmacs']:5.3f} GMACs  "
              + "  ".join(f"{m} {row[f'{m}_ms']:.1f} мс" for m in ALL_MODES if f"{m}_ms" in row), flush=True)
    rows = list(old.values())
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.7, help="доля удаляемых каналов")
    ap.add_argument("--floor", type=int, default=8, help="минимум каналов в группе")
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3, help="шум вокруг подсказки")
    ap.add_argument("--recalib-images", type=int, default=200)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--greg-epochs", type=int, default=10)
    ap.add_argument("--greg-lr", type=float, default=1e-4)
    ap.add_argument("--greg-lam", type=float, default=0.05, help="макс. затухание за шаг")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон")
    args = ap.parse_args()
    if args.smoke:
        args.candidates, args.top, args.ft_epochs, args.greg_epochs, args.batch_size = 4, 2, 1, 1, 4

    X = args.level
    tag = f"{int(round(X * 100))}"
    seed_everything(42)
    device = get_device(args.device)
    models_dir, results_dir = out_dirs(args.smoke)
    real_models = out_dirs(False)[0]
    base_path, unstr_path = real_models / "base.pth", real_models / "unstructured_l1_90.pth"
    data = load_data(device, args.smoke)
    log_path = results_dir / "own_ft_log.jsonl"
    ex_cpu = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    ex_dev = ex_cpu.to(device)
    t_start = time.time()

    base_cpu = load_model(base_path, "cpu")
    base_stats = model_stats(base_cpu)
    base_info = {"nonzero_params": base_stats["nonzero_params"], "gmacs": base_stats["gmacs"]}
    _, groups = prunable_groups(base_cpu, ex_cpu)
    names = [n for n, _ in groups]
    C = np.array([g[0].dep.target.module.out_channels for _, g in groups])
    TOT = int(C.sum())
    uni = copy.deepcopy(base_cpu)
    tp.pruner.MetaPruner(uni, ex_cpu, importance=tp.importance.RandomImportance(), pruning_ratio=X,
                         ignored_layers=[uni.segmentation_head]).step()
    u = model_stats(uni)
    u_gmacs, u_params, u_ch = u["gmacs"], u["params"], count_channels(uni, ex_cpu)
    print(f"групп {len(C)}, каналов {TOT} | равномерный {X:.0%}: {u_gmacs:.3f} GMACs, {u_params / 1e6:.2f} M, "
          f"каналов {u_ch} (удалено {1 - u_ch / TOT:.1%})", flush=True)

    # подсказка B
    q = prior_keep(base_cpu, unstr_path, ex_cpu)
    shape0 = np.sqrt(np.array([max(q[n], 1e-3) for n in names]))  # веса ~ каналы^2
    target_keep = int(round((1 - X) * TOT))
    print("подсказка B (доля ненулевых весов у unstructured_l1_90):")
    for n, c in zip(names, C):
        print(f"  {c:4d}  {q[n]:.3f}  {n}")

    # кандидаты
    rng = np.random.default_rng(0)
    order = l1_order(base_cpu, ex_cpu)
    def fits(g, p):
        return g <= u_gmacs * 1.0005 and p <= u_params * 1.0005

    cands = []
    for i in range(args.candidates):
        shape = shape0 if i == 0 else shape0 * np.exp(rng.normal(0, args.noise, len(C)))
        # если не влезает по gmacs/весам, уменьшаем keep
        lo, hi = int(np.minimum(C, args.floor).sum()), target_keep
        keep = channels_for(shape, C, args.floor, hi)
        g, p = cost_of(base_cpu, ex_cpu, names, C, keep)
        if not fits(g, p):
            while hi - lo > 4:
                mid = (lo + hi) // 2
                keep_mid = channels_for(shape, C, args.floor, mid)
                if fits(*cost_of(base_cpu, ex_cpu, names, C, keep_mid)):
                    lo = mid
                else:
                    hi = mid
            keep = channels_for(shape, C, args.floor, lo)
            g, p = cost_of(base_cpu, ex_cpu, names, C, keep)
        ok = fits(g, p) and 1 - keep.sum() / TOT >= X - 0.002
        cands.append({"id": i, "keep": keep, "gmacs": g, "params": p, "removed": 1 - keep.sum() / TOT,
                      "feasible": ok, "quick_val_dice": None, "ft_val_dice": None})
        print(f"  кандидат {i:3d}: удалено каналов {1 - keep.sum() / TOT:.1%}, GMACs {g / u_gmacs:.2f}× равном., "
              f"веса {p / u_params:.2f}× равном.{'' if ok else '  (не проходит)'}", flush=True)
    feas = [c for c in cands if c["feasible"]]
    print(f"кандидатов {len(cands)}, проходят по цене {len(feas)}", flush=True)
    if not feas:
        sys.exit("нет подходящих кандидатов")

    # быстрая оценка
    base_dev = load_model(base_path, device)
    for c in feas:
        m = copy.deepcopy(base_dev)
        apply_plan(m, ex_dev, remove_plan(order, names, C, c["keep"]))
        recalibrate_bn(m, data["train"], args.recalib_images)
        c["quick_val_dice"] = evaluate(m, data["val"])["dice"]
        print(f"  кандидат {c['id']:3d}: {c['gmacs']:.3f} GMACs ({c['gmacs'] / u_gmacs:.2f}× равном.), "
              f"веса {c['params'] / u_params:.2f}× равном., быстрый Dice val {c['quick_val_dice']:.4f}", flush=True)
        del m
    feas.sort(key=lambda c: -c["quick_val_dice"])

    # дообучаем топ
    best, best_model = None, None
    for c in feas[:args.top]:
        seed_everything(0)
        m = copy.deepcopy(base_dev)
        apply_plan(m, ex_dev, remove_plan(order, names, C, c["keep"]))
        recalibrate_bn(m, data["train"])
        train(m, data["train"], data["val"], args.ft_epochs, args.ft_lr, args.batch_size, warmup_epochs=0,
              log_path=log_path, tag=f"own_b_{tag}_cand{c['id']}")
        c["ft_val_dice"] = evaluate(m, data["val"])["dice"]
        print(f"  кандидат {c['id']}: Dice val после дообучения {c['ft_val_dice']:.4f}", flush=True)
        if best is None or c["ft_val_dice"] > best["ft_val_dice"]:
            best, best_model = c, m
    print(f"победитель: кандидат {best['id']}", flush=True)

    with open(results_dir / f"own_search_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "feasible", "gmacs", "params", "removed_channels", "quick_val_dice", "ft_val_dice"] + names)
        for c in cands:
            w.writerow([c["id"], c["feasible"], round(c["gmacs"], 5), c["params"], round(c["removed"], 5),
                        c["quick_val_dice"], c["ft_val_dice"]] + c["keep"].tolist())

    results_csv = results_dir / "own_results.csv"
    # победитель без greg
    nog_file = f"models/own_b_{tag}.pth"
    m0 = copy.deepcopy(base_dev)
    apply_plan(m0, ex_dev, remove_plan(order, names, C, best["keep"]))
    recalibrate_bn(m0, data["train"])
    row, _ = measure(m0, data, "own_b", X, "pruned", base_info)
    append_row(results_csv, row)
    row, plain = measure(best_model, data, "own_b", X, "finetuned", base_info, model_file=nog_file)
    save_model(plain, models_dir.parent / nog_file, method="own_b", ratio=X, keep=best["keep"].tolist())
    append_row(results_csv, row)
    del m0, best_model

    # greg
    seed_everything(0)
    remove = remove_plan(order, names, C, best["keep"])
    m = copy.deepcopy(base_dev)
    targets = greg_targets(m, ex_dev, remove)
    t0 = time.time()
    greg_phase(m, data, targets, args.greg_epochs, args.greg_lr, args.greg_lam, batch_size=args.batch_size,
               log_path=log_path, tag=f"own_b_greg_{tag}")
    before = evaluate(m, data["val"])["dice"]
    apply_plan(m, ex_dev, remove)
    after = evaluate(m, data["val"])["dice"]
    recalibrate_bn(m, data["train"])
    after_bn = evaluate(m, data["val"])["dice"]
    greg_min = round((time.time() - t0) / 60, 1)
    print(f"GReg: Dice val до удаления {before:.4f}, сразу после удаления {after:.4f}, после пересчета BN {after_bn:.4f}",
          flush=True)
    row, _ = measure(m, data, "own_b_greg", X, "pruned", base_info, prune_sec=greg_min * 60)
    append_row(results_csv, row)
    t0 = time.time()
    seed_everything(0)
    train(m, data["train"], data["val"], args.ft_epochs, args.ft_lr, args.batch_size, warmup_epochs=0,
          log_path=log_path, tag=f"own_b_greg_{tag}")
    ft_min = round((time.time() - t0) / 60, 1)
    greg_file = f"models/own_b_greg_{tag}.pth"
    row, plain = measure(m, data, "own_b_greg", X, "finetuned", base_info,
                         prune_sec=greg_min * 60, ft_min=ft_min, model_file=greg_file)
    save_model(plain, models_dir.parent / greg_file, method="own_b_greg", ratio=X, keep=best["keep"].tolist())
    append_row(results_csv, row)
    print(f"-> {results_csv}", flush=True)

    if not args.no_bench:
        print("замеры времени:", flush=True)
        kw = {"runs": 5, "warmup": 2} if args.smoke else {}
        benchmark_models([models_dir.parent / nog_file, models_dir.parent / greg_file],
                         models_dir.parent / "onnx", results_dir / "own_benchmark.csv", **kw)
        print(f"-> {results_dir / 'own_benchmark.csv'}")
    print(f"всего {(time.time() - t_start) / 60:.1f} мин")


if __name__ == "__main__":
    main()
