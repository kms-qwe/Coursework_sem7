"""Свой метод прунинга: поиск распределения каналов по группам + GReg, сравнение с бейзлайном по нескольким зернам.

Результаты в runs/results/own/, модели в runs/models/own/. Посчитанное при перезапуске пропускается.
Скорость меряется отдельно (--bench-only).
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (IMG_SIZE, evaluate, get_device, load_data, load_model, model_stats, out_dirs, save_model,
                    seed_everything, torch, tp, train)
from own_method_mini import (apply_plan, channels_for, cost_of, greg_phase, greg_targets, l1_order, prior_keep,
                             prunable_groups, recalibrate_bn, remove_plan)
from prune_baseline import prune_structured, prune_unstructured, without_masks

FIELDS = ["method", "level", "seed", "stage", "params", "nonzero_params", "gmacs", "params_frac", "gmacs_frac",
          "removed_channels", "val_dice", "val_iou", "test_dice", "test_iou", "test_jac65", "test_dice_global",
          "ph2_dice", "ph2_iou", "greg_val_before", "search_min", "prune_sec", "greg_min", "ft_min", "model_file",
          "device"]
BASELINE_METHODS = ["random", "l1", "fpgm", "bn_scale", "taylor", "unstructured_l1"]
OWN_METHODS = ["own_a", "own_b", "own_b_greg", "uniform_greg"]


def pct(x):
    return int(round(x * 100))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def has_masks(model):
    return any(n.endswith("weight_mask") for n, _ in model.named_buffers())


class Results:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if path.exists():
            with open(path) as f:
                self.done = {(r["method"], r["level"], r["seed"], r["stage"]) for r in csv.DictReader(f)}

    def has(self, method, level, seed, stage):
        return (method, str(level), str(seed), stage) in self.done

    def add(self, row):
        key = (row["method"], str(row["level"]), str(row["seed"]), row["stage"])
        if key in self.done:
            return
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        self.done.add(key)
        print(f"  -> {row['method']} {row['level']}% зерно {row['seed']} {row['stage']:9s}: "
              f"удалено каналов {row['removed_channels']:.1%}, GMACs {row['gmacs']:.3f}, "
              f"веса {row['nonzero_params'] / 1e6:.2f} M | Dice val {row['val_dice']:.4f}", flush=True)


class Ctx:
    def __init__(self, args, need_data=True):
        self.args = args
        self.device = get_device(args.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        models_dir, results_dir = out_dirs(args.smoke)
        self.runs_dir = models_dir.parent
        self.models_dir, self.results_dir = models_dir / "own", results_dir / "own"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.real_models = out_dirs(False)[0]
        self.base_path = self.real_models / "base.pth"
        self.results = Results(self.results_dir / "own_results.csv")
        self.log_path = self.results_dir / "own_ft_log.jsonl"
        if not need_data:
            return

        prior_path = self.real_models / f"{args.prior}.pth"
        for p in (self.base_path, prior_path):
            if not p.exists():
                sys.exit(f"нет {p}")
        self.data = load_data(self.device, args.smoke)
        self.ex_cpu = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        self.ex_dev = self.ex_cpu.to(self.device)
        self.base_cpu = load_model(self.base_path, "cpu")
        self.base_dev = load_model(self.base_path, self.device)
        s = model_stats(self.base_cpu)
        self.base_info = {"nonzero_params": s["nonzero_params"], "gmacs": s["gmacs"]}

        _, groups = prunable_groups(self.base_cpu, self.ex_cpu)
        self.names = [n for n, _ in groups]
        self.C = np.array([g[0].dep.target.module.out_channels for _, g in groups])
        self.TOT = int(self.C.sum())
        self.order = l1_order(self.base_cpu, self.ex_cpu)
        q = prior_keep(self.base_cpu, prior_path, self.ex_cpu)
        self.prior_shape = np.sqrt(np.array([max(q[n], 1e-3) for n in self.names]))

    def channels(self, model):
        mods = dict(model.named_modules())
        return sum(mods[n].out_channels for n in self.names)

    def model_rel(self, method, level, seed):
        return f"models/own/{method}_{level}_s{seed}.pth"

    def cut(self, keep):
        m = copy.deepcopy(self.base_dev)
        apply_plan(m, self.ex_dev, remove_plan(self.order, self.names, self.C, keep))
        return m

    def pruned(self, keep, n_images=None):
        m = self.cut(keep)
        recalibrate_bn(m, self.data["train"], n_images)
        return m

    def finetune(self, model, seed, tag):
        a = self.args
        seed_everything(seed)
        t0 = time.time()
        train(model, self.data["train"], self.data["val"], a.ft_epochs, a.ft_lr, a.batch_size, warmup_epochs=0,
              log_path=self.log_path, tag=tag)
        return (time.time() - t0) / 60

    def measure(self, model, method, level, seed, stage, **extra):
        plain = without_masks(model) if has_masks(model) else model
        stats = model_stats(plain)
        ev = {s: evaluate(model, self.data[s]) for s in ("val", "test", "ph2")}
        row = {"method": method, "level": level, "seed": seed, "stage": stage, **stats,
               "params_frac": stats["nonzero_params"] / self.base_info["nonzero_params"],
               "gmacs_frac": stats["gmacs"] / self.base_info["gmacs"],
               "removed_channels": 1 - self.channels(plain) / self.TOT,
               "val_dice": ev["val"]["dice"], "val_iou": ev["val"]["iou"],
               "test_dice": ev["test"]["dice"], "test_iou": ev["test"]["iou"], "test_jac65": ev["test"]["jac65"],
               "test_dice_global": ev["test"]["dice_global"],
               "ph2_dice": ev["ph2"]["dice"], "ph2_iou": ev["ph2"]["iou"], "device": self.device.type, **extra}
        self.results.add(row)
        return plain

    def save(self, model, method, level, seed, **meta):
        rel = self.model_rel(method, level, seed)
        save_model(model, self.runs_dir / rel, method=method, level=level, seed=seed, **meta)
        return rel

    def free(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_state(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path, st):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    tmp.replace(path)


def uniform_ref(ctx, X):
    uni = copy.deepcopy(ctx.base_cpu)
    tp.pruner.MetaPruner(uni, ctx.ex_cpu, importance=tp.importance.RandomImportance(), pruning_ratio=X,
                         ignored_layers=[uni.segmentation_head]).step()
    s = model_stats(uni)
    mods = dict(uni.named_modules())
    keep = np.array([mods[n].out_channels for n in ctx.names])
    return {"gmacs": float(s["gmacs"]), "params": int(s["params"]), "keep": keep.tolist(),
            "removed": float(1 - keep.sum() / ctx.TOT)}


def make_candidates(ctx, X, floor, uni, variant, n, rng):
    C, TOT = ctx.C, ctx.TOT
    target_keep = int(round((1 - X) * TOT))

    def fits(g, p):
        return g <= uni["gmacs"] * 1.0005 and p <= uni["params"] * 1.0005

    def cost(keep):
        return cost_of(ctx.base_dev, ctx.ex_dev, ctx.names, C, keep)

    cands = []
    for i in range(n):
        if variant == "A":
            shape = rng.uniform(0.1, 1.0, len(C))
        else:
            shape = ctx.prior_shape if i == 0 else ctx.prior_shape * np.exp(rng.normal(0, ctx.args.noise, len(C)))
        lo, hi = int(np.minimum(C, floor).sum()), target_keep
        keep = channels_for(shape, C, floor, hi)
        g, p = cost(keep)
        if not fits(g, p):
            while hi - lo > 4:
                mid = (lo + hi) // 2
                if fits(*cost(channels_for(shape, C, floor, mid))):
                    lo = mid
                else:
                    hi = mid
            keep = channels_for(shape, C, floor, lo)
            g, p = cost(keep)
        removed = 1 - keep.sum() / TOT
        ok = bool(fits(g, p) and removed >= X - 0.002)
        cands.append({"id": i, "keep": [int(k) for k in keep], "gmacs": float(g), "params": int(p),
                      "removed": float(removed), "feasible": ok, "quick": None})
        print(f"  {variant}{i:3d}: удалено каналов {removed:.1%}, GMACs {g / uni['gmacs']:.2f}× равном., "
              f"веса {p / uni['params']:.2f}× равном.{'' if ok else '  (не проходит)'}", flush=True)
    ctx.free()
    return cands


def search_level(ctx, X, uni, st, state_path):
    a = ctx.args
    L = pct(X)
    floor = a.floor_90 if L >= 90 else a.floor
    for k in ("cands", "top", "ft", "best", "sec"):
        st.setdefault(k, {})
    st["uniform"] = uni

    for v in ("A", "B"):
        if v in st["cands"]:
            continue
        print(f"--- {L}%: кандидаты {v} (минимум {floor} каналов в группе)", flush=True)
        t0 = time.time()
        rng = np.random.default_rng(1000 * L + (0 if v == "A" else 1))
        cands = make_candidates(ctx, X, floor, uni, v, a.candidates, rng)
        feas = [c for c in cands if c["feasible"]]
        print(f"--- {L}%: проходят по цене {len(feas)} из {len(cands)}, быстрая оценка", flush=True)
        for c in feas:
            m = ctx.pruned(np.array(c["keep"]), a.recalib_images)
            c["quick"] = float(evaluate(m, ctx.data["val"])["dice"])
            print(f"  {v}{c['id']:3d}: удалено {c['removed']:.1%}, GMACs {c['gmacs']:.3f}, "
                  f"веса {c['params'] / 1e6:.2f} M, быстрый Dice val {c['quick']:.4f}", flush=True)
            del m
        ctx.free()
        if not feas:
            sys.exit(f"{L}%: ни один кандидат {v} не проходит по цене")
        feas.sort(key=lambda c: -c["quick"])
        st["cands"][v] = cands
        st["top"][v] = [c["id"] for c in feas[:a.top]]
        st["sec"][f"search_{v}"] = time.time() - t0
        save_state(state_path, st)

    by_id = {v: {c["id"]: c for c in st["cands"][v]} for v in ("A", "B")}

    for v in ("A", "B"):
        method = f"own_{v.lower()}"
        for cid in st["top"][v]:
            key = f"{v}:{cid}"
            if key in st["ft"]:
                continue
            c = by_id[v][cid]
            print(f"--- {L}%: дообучение {v}{cid} (быстрый Dice {c['quick']:.4f})", flush=True)
            m = ctx.pruned(np.array(c["keep"]))
            ft_min = ctx.finetune(m, 0, f"{method}_{L}_cand{cid}")
            d = float(evaluate(m, ctx.data["val"])["dice"])
            best = st["best"].get(v)
            if best is None or d > st["ft"][f"{v}:{best}"]:
                ctx.save(m, method, L, 0, cand=cid, keep=c["keep"], ft_min=ft_min)
                st["best"][v] = cid
            st["ft"][key] = d
            st["sec"][f"ft_{v}"] = st["sec"].get(f"ft_{v}", 0) + ft_min * 60
            save_state(state_path, st)
            print(f"  {v}{cid}: Dice val после дообучения {d:.4f} (лучший {v}{st['best'][v]})", flush=True)
            del m
            ctx.free()

    if L == pct(a.rank_level) and a.rank_extra > 0:
        if "extra" not in st:
            pool = sorted([(v, c) for v in ("A", "B") for c in st["cands"][v]
                           if c["feasible"] and c["id"] not in st["top"][v]], key=lambda vc: -vc[1]["quick"])
            idx = sorted(set(np.linspace(0, len(pool) - 1, a.rank_extra).round().astype(int).tolist())) if pool else []
            st["extra"] = [f"{pool[i][0]}:{pool[i][1]['id']}" for i in idx]
            save_state(state_path, st)
        for key in st["extra"]:
            if key in st["ft"]:
                continue
            v, cid = key.split(":")
            c = by_id[v][int(cid)]
            print(f"--- {L}%: проверка порядка, дообучение {v}{cid} (быстрый Dice {c['quick']:.4f})", flush=True)
            m = ctx.pruned(np.array(c["keep"]))
            ctx.finetune(m, 0, f"own_{v.lower()}_{L}_cand{cid}_rank")
            st["ft"][key] = float(evaluate(m, ctx.data["val"])["dice"])
            save_state(state_path, st)
            del m
            ctx.free()

    write_search_csv(ctx, L, st)

    for v in ("A", "B"):
        method = f"own_{v.lower()}"
        keep = np.array(by_id[v][st["best"][v]]["keep"])
        search_min = (st["sec"].get(f"search_{v}", 0) + st["sec"].get(f"ft_{v}", 0)) / 60
        if not ctx.results.has(method, L, 0, "pruned"):
            ctx.measure(ctx.cut(keep), method, L, 0, "pruned", search_min=search_min)
        if not ctx.results.has(method, L, 0, "pruned_bn"):
            ctx.measure(ctx.pruned(keep), method, L, 0, "pruned_bn", search_min=search_min)
        if not ctx.results.has(method, L, 0, "finetuned"):
            rel = ctx.model_rel(method, L, 0)
            ckpt_ft = torch.load(ctx.runs_dir / rel, map_location="cpu").get("ft_min")
            ctx.measure(load_model(ctx.runs_dir / rel, ctx.device), method, L, 0, "finetuned",
                        search_min=search_min, ft_min=ckpt_ft, model_file=rel)
        ctx.free()


def write_search_csv(ctx, L, st):
    path = ctx.results_dir / f"own_search_{L}.csv"
    uni = st["uniform"]
    pts = []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "id", "feasible", "gmacs", "params", "removed_channels", "quick_val_dice",
                    "ft_val_dice", "in_top", "winner"] + ctx.names)
        w.writerow(["U", -1, True, round(uni["gmacs"], 5), uni["params"], round(uni["removed"], 5), "", "",
                    False, False] + uni["keep"])
        for v in ("A", "B"):
            for c in st["cands"][v]:
                ft = st["ft"].get(f"{v}:{c['id']}")
                if ft is not None:
                    pts.append((c["quick"], ft))
                w.writerow([v, c["id"], c["feasible"], round(c["gmacs"], 5), c["params"], round(c["removed"], 5),
                            c["quick"], ft, c["id"] in st["top"][v], c["id"] == st["best"].get(v)] + c["keep"])
    rho = spearman([p[0] for p in pts], [p[1] for p in pts])
    print(f"--- {L}%: ранговая корреляция быстрый Dice / после дообучения: {rho:.3f} по {len(pts)} кандидатам "
          f"-> {path}", flush=True)


def baseline_run(ctx, method, X, seed):
    L = pct(X)
    if ctx.results.has(method, L, seed, "finetuned"):
        return
    a = ctx.args
    print(f"--- {L}%: {method}, зерно {seed}", flush=True)
    seed_everything(seed)
    m = load_model(ctx.base_path, ctx.device)
    t0 = time.time()
    if method == "unstructured_l1":
        prune_unstructured(m, X)
    else:
        prune_structured(m, method, X, ctx.data["train"], a.batch_size, a.taylor_batches)
    m.zero_grad(set_to_none=True)
    prune_sec = time.time() - t0
    plain = ctx.measure(m, method, L, seed, "pruned", prune_sec=prune_sec)
    bn = copy.deepcopy(plain)
    recalibrate_bn(bn, ctx.data["train"])
    ctx.measure(bn, method, L, seed, "pruned_bn", prune_sec=prune_sec)
    del bn, plain
    ft_min = ctx.finetune(m, seed, f"{method}_{L}_s{seed}")
    plain = without_masks(m) if has_masks(m) else m
    rel = ctx.save(plain, method, L, seed, ft_min=ft_min)
    ctx.measure(plain, method, L, seed, "finetuned", prune_sec=prune_sec, ft_min=ft_min, model_file=rel)
    del m, plain
    ctx.free()


def own_plan_run(ctx, method, L, seed, keep):
    if ctx.results.has(method, L, seed, "finetuned"):
        return
    print(f"--- {L}%: {method}, зерно {seed}", flush=True)
    keep = np.array(keep)
    m = ctx.cut(keep)
    ctx.measure(m, method, L, seed, "pruned")
    recalibrate_bn(m, ctx.data["train"])
    ctx.measure(m, method, L, seed, "pruned_bn")
    ft_min = ctx.finetune(m, seed, f"{method}_{L}_s{seed}")
    rel = ctx.save(m, method, L, seed, keep=keep.tolist(), ft_min=ft_min)
    ctx.measure(m, method, L, seed, "finetuned", ft_min=ft_min, model_file=rel)
    del m
    ctx.free()


def greg_run(ctx, method, L, seed, keep, search_min=None):
    if ctx.results.has(method, L, seed, "finetuned"):
        return
    a = ctx.args
    tag = f"{method}_{L}_s{seed}"
    print(f"--- {L}%: {method}, зерно {seed}", flush=True)
    remove = remove_plan(ctx.order, ctx.names, ctx.C, keep)
    seed_everything(seed)
    m = copy.deepcopy(ctx.base_dev)
    targets = greg_targets(m, ctx.ex_dev, remove)
    t0 = time.time()
    greg_phase(m, ctx.data, targets, a.greg_epochs, a.greg_lr, a.greg_lam, batch_size=a.batch_size,
               log_path=ctx.log_path, tag=tag)
    before = evaluate(m, ctx.data["val"])["dice"]
    apply_plan(m, ctx.ex_dev, remove)
    greg_min = (time.time() - t0) / 60
    extra = {"greg_val_before": before, "greg_min": greg_min, "search_min": search_min}
    ctx.measure(m, method, L, seed, "pruned", **extra)
    recalibrate_bn(m, ctx.data["train"])
    ctx.measure(m, method, L, seed, "pruned_bn", **extra)
    ft_min = ctx.finetune(m, seed, tag)
    rel = ctx.save(m, method, L, seed, keep=[int(k) for k in keep], ft_min=ft_min, greg_min=greg_min)
    ctx.measure(m, method, L, seed, "finetuned", ft_min=ft_min, model_file=rel, **extra)
    del m
    ctx.free()


def bench(ctx, levels):
    from own_method_mini import benchmark_models
    paths = [ctx.base_path]
    for X in levels:
        L = pct(X)
        paths += [ctx.models_dir / f"{m}_{L}_s0.pth" for m in BASELINE_METHODS + OWN_METHODS]
    out_csv = ctx.results_dir / "own_benchmark.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {r["model"] for r in csv.DictReader(f)}
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"нет моделей ({len(missing)}): {', '.join(missing)}")
    todo = [p for p in paths if p.exists() and p.stem not in done]
    print(f"===== замеры времени: {len(todo)} моделей -> {out_csv}", flush=True)
    if todo:
        kw = {"runs": 5, "warmup": 2} if ctx.args.smoke else {}
        benchmark_models(todo, ctx.runs_dir / "onnx" / "own", out_csv, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=float, default=[0.5, 0.7, 0.3, 0.9])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--baseline-methods", nargs="+", default=BASELINE_METHODS, choices=BASELINE_METHODS)
    ap.add_argument("--floor", type=int, default=8, help="минимум каналов в группе (30-70%%)")
    ap.add_argument("--floor-90", type=int, default=1, help="минимум каналов в группе на 90%%")
    ap.add_argument("--candidates", type=int, default=40, help="кандидатов в каждом варианте")
    ap.add_argument("--noise", type=float, default=0.3, help="шум вокруг подсказки B")
    ap.add_argument("--prior", default="unstructured_l1_90", help="модель-подсказка для B (runs/models/<имя>.pth)")
    ap.add_argument("--recalib-images", type=int, default=200, help="снимков для пересчета BN в быстрой оценке")
    ap.add_argument("--top", type=int, default=5, help="сколько кандидатов дообучать в каждом варианте")
    ap.add_argument("--rank-level", type=float, default=0.5, help="уровень проверки порядка кандидатов")
    ap.add_argument("--rank-extra", type=int, default=5, help="доп. кандидатов из середины/хвоста")
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--greg-epochs", type=int, default=10)
    ap.add_argument("--greg-lr", type=float, default=1e-4)
    ap.add_argument("--greg-lam", type=float, default=0.05, help="макс. затухание за шаг")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taylor-batches", type=int, default=16)
    ap.add_argument("--bench-only", action="store_true", help="только замеры скорости готовых моделей")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон на 16 картинках")
    args = ap.parse_args()
    if args.smoke:
        args.candidates, args.top, args.rank_extra, args.recalib_images = 3, 2, 1, 8
        args.ft_epochs, args.greg_epochs, args.batch_size, args.taylor_batches = 1, 1, 4, 2
        args.seeds = args.seeds[:2]

    t_start = time.time()
    if args.bench_only:
        bench(Ctx(args, need_data=False), args.levels)
        return

    seed_everything(42)
    ctx = Ctx(args)
    print(f"device {ctx.device} | групп {len(ctx.C)}, каналов {ctx.TOT} | уровни {[pct(x) for x in args.levels]} "
          f"| зерна {args.seeds} | -> {ctx.results_dir}", flush=True)
    if not ctx.results.has("base", 0, 0, "finetuned"):
        ctx.measure(ctx.base_dev, "base", 0, 0, "finetuned", model_file="models/base.pth")

    for X in args.levels:
        L = pct(X)
        t_level = time.time()
        print(f"===== уровень {L}%: поиск плана", flush=True)
        state_path = ctx.results_dir / f"own_state_{L}.json"
        st = load_state(state_path)
        uni = st.get("uniform") or uniform_ref(ctx, X)
        print(f"равномерный {L}%: {uni['gmacs']:.3f} GMACs, {uni['params'] / 1e6:.2f} M весов, "
              f"удалено каналов {uni['removed']:.1%}", flush=True)
        search_level(ctx, X, uni, st, state_path)
        keep_a = next(c for c in st["cands"]["A"] if c["id"] == st["best"]["A"])["keep"]
        keep_b = next(c for c in st["cands"]["B"] if c["id"] == st["best"]["B"])["keep"]
        search_min_b = (st["sec"].get("search_B", 0) + st["sec"].get("ft_B", 0)) / 60

        for s in args.seeds:
            print(f"===== уровень {L}%, зерно {s} ({(time.time() - t_start) / 60:.0f} мин от старта)", flush=True)
            for method in args.baseline_methods:
                baseline_run(ctx, method, X, s)
            if s != 0:
                own_plan_run(ctx, "own_a", L, s, keep_a)
                own_plan_run(ctx, "own_b", L, s, keep_b)
            greg_run(ctx, "own_b_greg", L, s, keep_b, search_min=search_min_b)
            greg_run(ctx, "uniform_greg", L, s, uni["keep"])
        print(f"===== уровень {L}% готов за {(time.time() - t_level) / 60:.1f} мин "
              f"(всего {(time.time() - t_start) / 60:.1f} мин)", flush=True)

    print(f"-> {ctx.results.path}")
    print(f"всего {(time.time() - t_start) / 60:.1f} мин")


if __name__ == "__main__":
    main()
