"""Отчет по своему методу: own_results.csv + own_benchmark.csv + own_search_*.csv -> own_summary.md и графики.

"Заметно" = разница средних больше двух стандартных ошибок.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import out_dirs

BASELINE = ["random", "l1", "fpgm", "bn_scale", "taylor", "unstructured_l1"]
STRUCTURED = BASELINE[:5]
OWN = ["own_a", "own_b", "own_b_greg", "uniform_greg"]
NAMES = {"own_a": "поиск A", "own_b": "поиск B", "own_b_greg": "**поиск B + GReg (наш)**",
         "uniform_greg": "равномерный + GReg"}
COLORS = {"random": "#2a78d6", "l1": "#eb6834", "fpgm": "#1baf7a", "bn_scale": "#eda100", "taylor": "#e87ba4",
          "own_b": "#6250d6", "own_b_greg": "#e34948"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


class Table:
    def __init__(self, rows):
        self.by = defaultdict(list)
        for r in rows:
            self.by[(r["method"], int(r["level"]), r["stage"])].append(r)

    def vals(self, method, L, stage, key):
        return [float(r[key]) for r in self.by.get((method, L, stage), []) if r.get(key) not in ("", None)]

    def mean(self, method, L, stage, key):
        v = self.vals(method, L, stage, key)
        return float(np.mean(v)) if v else None

    def ms(self, method, L, stage, key):
        v = self.vals(method, L, stage, key)
        if not v:
            return "—"
        if len(v) == 1:
            return f"{v[0]:.4f}"
        return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"

    def n(self, method, L, stage="finetuned"):
        return len(self.by.get((method, L, stage), []))

    def first(self, method, L, stage, key):
        rs = self.by.get((method, L, stage), [])
        return rs[0].get(key) if rs else None


def delta(t, a, b, L, key="test_dice"):
    va, vb = t.vals(a, L, "finetuned", key), t.vals(b, L, "finetuned", key)
    if not va or not vb:
        return "—"
    d = np.mean(vb) - np.mean(va)
    if len(va) < 2 or len(vb) < 2:
        return f"{d:+.4f}"
    se = np.sqrt(np.var(va, ddof=1) / len(va) + np.var(vb, ddof=1) / len(vb))
    mark = "заметно" if abs(d) > 2 * se else "в пределах шума"
    return f"{d:+.4f} ({mark})"


def f(v, digits=4):
    return "—" if v in (None, "") else f"{float(v):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _, results_dir = out_dirs(args.smoke)
    own_dir = results_dir / "own"
    rows = read_csv(own_dir / "own_results.csv")
    if not rows:
        sys.exit(f"нет {own_dir / 'own_results.csv'}, сначала own_method.py")
    t = Table(rows)
    bench = {r["model"]: r for r in read_csv(own_dir / "own_benchmark.csv")}
    levels = sorted({int(r["level"]) for r in rows if int(r["level"]) > 0})
    seeds = sorted({int(r["seed"]) for r in rows})
    device = ", ".join(sorted({r.get("device", "") for r in rows if r.get("device")}))

    out = ["# Свой метод: результаты", "",
           f"ISIC 2016 test (379) и PH2 (200), 256x256. Обучение на {device or '?'}, зерна {seeds}; "
           "Dice — среднее ± ст. откл. по зернам. Выбор кандидатов и победителей только по val. "
           "Время — медиана, batch 1, модели зерна 0, замер локально.", ""]

    out += ["## 1. Все методы", "",
            "Стадии: «после прунинга» — сразу после удаления каналов (у GReg — после гашения и удаления); "
            "«+ пересчет BN» — то же после пересчета статистик BN (у бейзлайна на копии, дообучение идет без него); "
            "«после дообучения» — итог. У unstructured_l1 веса — ненулевые, GMACs не меняются.", "",
            "| Уровень | Метод | Зерен | Удалено каналов | Веса, M | GMACs | Dice test после прунинга "
            "| + пересчет BN | Dice test после дообучения | IoU test | Dice PH2 | Файл, МБ | CPU 1п, мс | ORT 1п, мс |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    b = bench.get("base", {})
    out.append(f"| 0% | исходная | — | 0% | {t.mean('base', 0, 'finetuned', 'nonzero_params') / 1e6:.2f} "
               f"| {f(t.mean('base', 0, 'finetuned', 'gmacs'), 2)} | — | — | {t.ms('base', 0, 'finetuned', 'test_dice')} "
               f"| {t.ms('base', 0, 'finetuned', 'test_iou')} | {t.ms('base', 0, 'finetuned', 'ph2_dice')} "
               f"| {f(b.get('file_mb'), 1)} | {f(b.get('cpu1_ms'), 1)} | {f(b.get('ort1_ms'), 1)} |"
               if t.n("base", 0) else "| 0% | исходная | — | | | | | | | | | | | |")
    for L in levels:
        for m in BASELINE + OWN:
            if not t.n(m, L):
                continue
            bb = bench.get(f"{m}_{L}_s0", {})
            name = NAMES.get(m, m)
            out.append(
                f"| {L}% | {name} | {t.n(m, L)} | {t.mean(m, L, 'finetuned', 'removed_channels'):.1%} "
                f"| {t.mean(m, L, 'finetuned', 'nonzero_params') / 1e6:.2f} | {t.mean(m, L, 'finetuned', 'gmacs'):.3f} "
                f"| {t.ms(m, L, 'pruned', 'test_dice')} | {t.ms(m, L, 'pruned_bn', 'test_dice')} "
                f"| {t.ms(m, L, 'finetuned', 'test_dice')} | {t.ms(m, L, 'finetuned', 'test_iou')} "
                f"| {t.ms(m, L, 'finetuned', 'ph2_dice')} | {f(bb.get('file_mb'), 1)} | {f(bb.get('cpu1_ms'), 1)} "
                f"| {f(bb.get('ort1_ms'), 1)} |")
    out.append("")

    out += ["## 2. Поиск B + GReg против каждого метода бейзлайна", "",
            "Разница средних Dice после дообучения (наш минус метод). «Заметно» — больше двух стандартных ошибок.", "",
            "| Уровень | Метод | Δ Dice test | Δ Dice PH2 | Δ CPU 1п, мс |", "|---|---|---|---|---|"]
    for L in levels:
        if not t.n("own_b_greg", L):
            continue
        for m in BASELINE:
            if not t.n(m, L):
                continue
            to, tm = bench.get(f"own_b_greg_{L}_s0", {}).get("cpu1_ms"), bench.get(f"{m}_{L}_s0", {}).get("cpu1_ms")
            dt = f"{float(to) - float(tm):+.1f}" if to and tm else "—"
            out.append(f"| {L}% | {m} | {delta(t, m, 'own_b_greg', L)} | {delta(t, m, 'own_b_greg', L, 'ph2_dice')} "
                       f"| {dt} |")
    out.append("")

    out += ["## 3. Абляция (Dice test после дообучения)", "",
            "| Уровень | равномерный (l1) | равномерный + GReg | поиск A | поиск B | поиск B + GReg |",
            "|---|---|---|---|---|---|"]
    comp = []
    for L in levels:
        if not t.n("own_b_greg", L):
            continue
        out.append(f"| {L}% | " + " | ".join(t.ms(m, L, "finetuned", "test_dice")
                                            for m in ("l1", "uniform_greg", "own_a", "own_b", "own_b_greg")) + " |")
        comp.append(f"| {L}% | {delta(t, 'l1', 'own_a', L)} | {delta(t, 'own_a', 'own_b', L)} "
                    f"| {delta(t, 'own_b', 'own_b_greg', L)} | {delta(t, 'l1', 'uniform_greg', L)} |")
    out += ["", "| Уровень | 1. неравномерность (A − равномерный) | 2. подсказка (B − A) "
            "| 3. GReg с поиском (B+GReg − B) | 4. GReg без поиска (равн.+GReg − равн.) |",
            "|---|---|---|---|---|"] + comp + [""]

    out += ["## 4. Порядок кандидатов: быстрый Dice (пересчет BN) против Dice после дообучения (val)", "",
            "| Уровень | Кандидатов дообучено | Спирмен | Лучший по быстрому = лучший после дообучения "
            "| Победитель B: Dice val | Лучший из всех дообученных |", "|---|---|---|---|---|---|"]
    from own_method import spearman
    for L in levels:
        rs = [r for r in read_csv(own_dir / f"own_search_{L}.csv") if r["ft_val_dice"] not in ("", None)]
        if not rs:
            continue
        q = np.array([float(r["quick_val_dice"]) for r in rs])
        ft = np.array([float(r["ft_val_dice"]) for r in rs])
        win = next((float(r["ft_val_dice"]) for r in rs if r["variant"] == "B" and r["winner"] == "True"), None)
        out.append(f"| {L}% | {len(rs)} | {spearman(q, ft):.3f} | {'да' if np.argmax(q) == np.argmax(ft) else 'нет'} "
                   f"| {f(win)} | {ft.max():.4f} |")
    out.append("")

    out += ["## 5. GReg: Dice val до и после удаления каналов (среднее по зернам)", "",
            "| Уровень | Метод | до удаления | сразу после удаления | + пересчет BN | без GReg: после удаления + BN |",
            "|---|---|---|---|---|---|"]
    for L in levels:
        for m, ref in (("own_b_greg", "own_b"), ("uniform_greg", "l1")):
            if t.n(m, L, "pruned"):
                out.append(f"| {L}% | {m} | {f(t.mean(m, L, 'pruned', 'greg_val_before'))} "
                           f"| {f(t.mean(m, L, 'pruned', 'val_dice'))} | {f(t.mean(m, L, 'pruned_bn', 'val_dice'))} "
                           f"| {f(t.mean(ref, L, 'pruned_bn', 'val_dice'))} ({ref}) |")
    out.append("")

    plots, notes = make_plots(own_dir, t, levels)
    if plots:
        out += ["## 6. Графики", "", "Точка — среднее по зернам, планка — ст. откл. unstructured_l1 не показан: "
                "GMACs у него не меняются.", ""]
        for p in plots:
            out.append(f"![{p.stem}]({p.name})")
            if notes.get(p.name):
                out.append(f"\nНе показаны (средний Dice < 0,5): {', '.join(notes[p.name])}.")
            out.append("")

    path = own_dir / "own_summary.md"
    path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"-> {path}")


def make_plots(own_dir, t, levels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, графики пропущены (pip install matplotlib)")
        return [], {}
    import matplotlib.ticker
    paths, notes = [], {}
    for key, title, fname in [("test_dice", "Dice на test (ISIC 2016)", "dice_test_vs_gmacs.png"),
                              ("ph2_dice", "Dice на PH2 (внешний тест)", "dice_ph2_vs_gmacs.png")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        lows, xs = [], []
        for m in STRUCTURED + ["own_b", "own_b_greg"]:
            pts = []
            for L in levels:
                v = t.vals(m, L, "finetuned", key)
                if not v:
                    continue
                g = t.mean(m, L, "finetuned", "gmacs")
                if np.mean(v) < 0.5:
                    lows.append(f"{m} {L}%")
                    continue
                pts.append((g, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0))
            if not pts:
                continue
            pts.sort()
            xs += [p[0] for p in pts]
            ours = m.startswith("own")
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=COLORS[m],
                        lw=2 if ours else 1.5, marker="D" if m == "own_b_greg" else ("s" if ours else "o"),
                        ms=7 if ours else 4, capsize=3, label=NAMES.get(m, m).strip("*"),
                        markeredgecolor=SURFACE if ours else COLORS[m], markeredgewidth=1.2 if ours else 0)
        base = t.mean("base", 0, "finetuned", key)
        if base:
            ax.axhline(base, color=INK2, lw=1, ls="--")
            ax.text(0.98, base, " исходная", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("GMACs (лог. шкала)", color=INK2)
        ax.set_ylabel("Dice", color=INK2)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ticks = []
        for x in sorted(float(f"{x:.2g}") for x in xs):
            if not ticks or x / ticks[-1] > 1.1:
                ticks.append(x)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{g:g}" for g in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        notes[fname] = lows
        p = own_dir / fname
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
        print(f"-> {p}")
    return paths, notes


if __name__ == "__main__":
    main()
