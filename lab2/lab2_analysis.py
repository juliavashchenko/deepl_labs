from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions, split_summary

FN_PRICE = 10
FP_PRICE = 1
COLS = ["threshold", "TP", "FP", "FN", "TN", "C"]

SMALL_SETS = {
    "A": ([0, 1, 0, 1, 0, 0], [0.9, 0.7, 0.7, 0.4, 0.2, 0.1], np.float32(0.4)),
    "B": ([0, 0, 0], [0.9, 0.6, 0.3], np.inf),
    "C": ([1, 1] + [0] * 11, [0.8] + [0.5] * 11 + [0.2], np.float32(0.8)),
}
WHY = {
    "A": "рівні оцінки 0.7 (мітки 1 і 0) мають отримати однаковий прогноз: при t = 0.7 одразу TP = 1, FP = 2",
    "B": "позитивних міток немає: FN = 0 завжди, C = FP, мінімум 0 лише на граничному кандидаті +inf",
    "C": "C(0.8) = C(0.5) = 10 (один пропуск = десять хибних тривог): за правилом нічиєї беремо більший поріг 0.8",
}


def count_row(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    flagged = s >= t
    fraud = y == 1
    tp = np.sum(flagged & fraud)
    fp = np.sum(flagged & ~fraud)
    fn = np.sum(~flagged & fraud)
    tn = np.sum(~flagged & ~fraud)
    return {"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": FN_PRICE * fn + FP_PRICE * fp}


def rates(r: dict) -> dict:
    tp, fp, fn, tn = r["TP"], r["FP"], r["FN"], r["TN"]
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }


def compare_rules(y: np.ndarray, s: np.ndarray, t_star: float) -> pd.DataFrame:
    rows = []
    for rule, t in [("Стандартний поріг", 0.5), ("Мінімум вартості (t*)", t_star), ("Усі негативні", np.inf)]:
        r = count_row(y, s, t)
        rows.append({"rule": rule, **r, **rates(r)})
    return pd.DataFrame(rows)


def fast_search(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    order = np.argsort(-s, kind="stable")
    s_desc, y_desc = s[order], y[order]
    ends = np.flatnonzero(np.r_[s_desc[1:] != s_desc[:-1], True])

    tp = np.r_[0, np.cumsum(y_desc)[ends]]
    fp = np.r_[0, np.cumsum(1 - y_desc)[ends]]
    threshold = np.r_[np.inf, s_desc[ends]]
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {"threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": FN_PRICE * fn + FP_PRICE * fp}
    )


def slow_search(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    candidates = np.r_[np.inf, np.unique(s)[::-1]]
    return pd.DataFrame([count_row(y, s, t) for t in candidates])


def pick_threshold(table: pd.DataFrame) -> float:
    best = table["C"].min()
    return table.loc[table["C"] == best, "threshold"].max()


def draw_curves(y: np.ndarray, s: np.ndarray, auc: float, points: list, path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(rec, prec)
    ax1.set(xlabel="Recall", ylabel="Precision", title="PR-крива (validation)")
    ax2.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], ":", color="gray")
    ax2.set(xlabel="FPR", ylabel="TPR (recall)", title="ROC-крива (validation)")
    for r, mark, name in points:
        if not np.isnan(r["precision"]):
            ax1.scatter(r["recall"], r["precision"], marker=mark, s=80, label=name)
        ax2.scatter(r["fpr"], r["recall"], marker=mark, s=80, label=name)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step_validation(out: Path, data: dict) -> None:
    y = data["validation_y"]
    s = data["validation_scores"]

    table = fast_search(y, s)
    table.to_csv(out / "validation_candidates.csv", index=False)
    t_star = pick_threshold(table)
    ties = (table["C"] == table["C"].min()).sum()

    comparison = compare_rules(y, s, t_star)
    comparison.to_csv(out / "validation_rule_comparison.csv", index=False, na_rep="nan")

    star = comparison.loc[1]
    fast_row = table.loc[table["threshold"] == t_star].iloc[0]
    assert all(star[k] == fast_row[k] for k in COLS[1:]), "швидка таблиця розходиться з прямим підрахунком"

    auc = roc_auc_score(y, s)
    draw_curves(
        y, s, auc,
        [(comparison.loc[0], "s", "t = 0.5"), (star, "*", f"t* = {t_star:.4g}")],
        out / "validation_pr_roc.png",
    )
    pd.DataFrame(
        [{"t_star": t_star, "validation_C": star["C"], "roc_auc_validation": auc, "tied_thresholds": ties}]
    ).to_csv(out / "threshold.csv", index=False)

    print("\n=== КРОК 1. validation ===")
    print(f"ROC-AUC = {auc:.6f};  t* = {t_star!s};  C(t*) = {star['C']};  порогів з такою вартістю: {ties}")
    print(comparison.to_string(index=False))


def step_small_sets(out: Path) -> None:
    print("\n=== КРОК 2. перевірка на наборах A, B, C ===")
    tables, summary = [], []
    for name, (labels, scores, expected) in SMALL_SETS.items():
        y = np.array(labels)
        s = np.array(scores, dtype=np.float32)
        slow = slow_search(y, s)
        fast = fast_search(y, s)
        same = np.array_equal(fast.to_numpy(), slow.to_numpy())
        t_slow, t_fast = pick_threshold(slow), pick_threshold(fast)
        ok = same and t_slow == t_fast == expected
        print(f"\nНабір {name}  (оцінки {scores})")
        print(slow.to_string(index=False))
        print(f"t*: повільний = {t_slow!s}, швидкий = {t_fast!s}, очікуваний = {expected!s}; таблиці збігаються: {same} -> {'OK' if ok else 'ПОМИЛКА'}")
        tables.append(slow.assign(set=name))
        summary.append(
            {
                "set": name,
                "t_star_slow": str(t_slow),
                "t_star_fast": str(t_fast),
                "t_star_expected": str(expected),
                "tables_equal": same,
                "ok": ok,
                "what_it_checks": WHY[name],
            }
        )
    pd.concat(tables)[["set", *COLS]].to_csv(out / "small_sets_candidates.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "small_sets_summary.csv", index=False)
    if not all(r["ok"] for r in summary):
        raise SystemExit("Перевірка не пройдена: до test не переходимо.")
    print("\nУсі три перевірки пройдено.")


def step_test(out: Path, data: dict) -> None:
    t = np.float32(pd.read_csv(out / "threshold.csv").loc[0, "t_star"])
    y = data["test_y"]
    s = data["test_scores"]
    amount = data["test_amount"]
    ids = data["test_row_ids"]

    comparison = compare_rules(y, s, t)
    comparison.to_csv(out / "test_rule_comparison.csv", index=False, na_rep="nan")
    star = comparison.loc[1]

    flagged = s >= t
    order = np.argsort(ids)
    rows = []
    for kind, mask in [("FP", flagged & (y == 0)), ("FN", ~flagged & (y == 1))]:
        for p in order[mask[order]][:5]:
            rows.append(
                {
                    "csv_row": ids[p],
                    "type": kind,
                    "label": y[p],
                    "score": s[p],
                    "amount": amount[p],
                    "distance_to_t_star": abs(s[p] - t),
                }
            )
    errors = pd.DataFrame(
        rows, columns=["csv_row", "type", "label", "score", "amount", "distance_to_t_star"]
    )
    errors.to_csv(out / "test_errors.csv", index=False)

    print(f"\n=== КРОК 3. test (t* = {t!s}) ===")
    print(comparison.to_string(index=False))
    if np.isnan(star["precision"]):
        print("precision не визначена: TP + FP = 0, нічого не позначено шахрайством")
    if np.isnan(star["recall"]):
        print("recall не визначений: TP + FN = 0, у test немає шахрайств")
    print("\nПерші п'ять FP та FN за номером рядка CSV:")
    print(errors.to_string(index=False))
    base = comparison.loc[0]
    print(
        f"\nВідносно порога 0.5: FP {star['FP'] - base['FP']:+d}, "
        f"FN {star['FN'] - base['FN']:+d}, C {star['C'] - base['C']:+d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ЛР2, пункти 2-4: аналіз порога зі збережених прогнозів")
    ap.add_argument("--saved", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    if not (args.saved / "predictions.npz").exists():
        ap.error(f"у {args.saved} немає predictions.npz: спочатку запустіть навчання")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.saved)
    print(split_summary(data).to_string(index=False))
    step_validation(args.out, data)
    step_small_sets(args.out)
    step_test(args.out, data)
    print(f"\nГотово. Результати: {args.out.resolve()}")


if __name__ == "__main__":
    main()
