from __future__ import annotations

import argparse
import shutil
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = REPO_ROOT / "app" / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
HIST_DIR = REPORTS_DIR / "histories"


def _run_command(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _train_model(model: str, seed: int) -> None:
    if model == "vgg16_cbam":
        _run_command([sys.executable, "app/ml/train_vgg16_cbam.py", "--seed", str(seed)])
        return
    if model == "resnet50":
        _run_command([sys.executable, "app/ml/train_resnet50.py", "--seed", str(seed)])
        return
    if model == "resnet50_cbam":
        _run_command([sys.executable, "app/ml/train_resnet50.py", "--use-cbam", "--seed", str(seed)])
        return
    raise ValueError(f"Model không hỗ trợ: {model}")


def _weights_history_from_model(model: str) -> tuple[Path, Path]:
    if model == "vgg16_cbam":
        return CHECKPOINT_DIR / "vgg16_cbam_best.weights.h5", CHECKPOINT_DIR / "training_history.json"
    return CHECKPOINT_DIR / f"{model}_best.weights.h5", CHECKPOINT_DIR / f"{model}_training_history.json"


def _snapshot_history(model: str, seed: int) -> Path | None:
    _, history = _weights_history_from_model(model)
    if not history.exists():
        return None
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    out = HIST_DIR / f"{model}_seed{seed}_training_history.json"
    shutil.copyfile(history, out)
    return out


def _evaluate_model(model: str, seed: int, min_recall_la_sau: float) -> Path:
    weights, history = _weights_history_from_model(model)
    prefix = f"{model}_seed{seed}_"
    _run_command(
        [
            sys.executable,
            "app/ml/generate_report_charts.py",
            "--model",
            model,
            "--weights",
            str(weights),
            "--history",
            str(history),
            "--reports-dir",
            str(REPORTS_DIR),
            "--fig-dir",
            str(FIG_DIR),
            "--prefix",
            prefix,
            "--seed",
            str(seed),
            "--min-recall-la-sau",
            str(min_recall_la_sau),
        ]
    )
    return REPORTS_DIR / f"{prefix}metrics_summary.json"


def _load_metrics(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    f1_la_sau = float(data["f1_per_class"]["la_sau"])
    macro_f1 = float(data["macro_f1"])
    tl_score = 0.65 * f1_la_sau + 0.35 * macro_f1
    return {
        "accuracy": float(data["accuracy"]),
        "macro_f1": macro_f1,
        "recall_la_sau": float(data["recall_la_sau"]),
        "precision_la_sau": float(data["precision_la_sau"]),
        "f1_la_sau": f1_la_sau,
        "tl_score": tl_score,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


def run_experiments(models: list[str], seeds: list[int], min_recall_la_sau: float, skip_train: bool) -> dict[str, object]:
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for model in models:
        model_runs: dict[str, dict[str, float]] = {}
        for seed in seeds:
            if not skip_train:
                _train_model(model, seed)
            hist_copy = _snapshot_history(model, seed)
            if hist_copy is not None:
                print(f"[SAVED] {hist_copy}")
            metrics_path = _evaluate_model(model, seed, min_recall_la_sau=min_recall_la_sau)
            run_metrics = _load_metrics(metrics_path)
            model_runs[str(seed)] = run_metrics
            print(f"[DONE] {model} seed={seed} => TL score={run_metrics['tl_score']:.4f}")

        all_results[model] = model_runs
        metric_names = list(next(iter(model_runs.values())).keys())
        summary[model] = {}
        for metric in metric_names:
            vals = [r[metric] for r in model_runs.values()]
            mean_v, std_v = _mean_std(vals)
            summary[model][metric] = {"mean": mean_v, "std": std_v}

    payload = {
        "models": models,
        "seeds": seeds,
        "min_recall_la_sau": min_recall_la_sau,
        "tl_score_formula": "0.65 * f1_la_sau + 0.35 * macro_f1",
        "per_seed_metrics": all_results,
        "summary_mean_std": summary,
    }
    return payload


def _format_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    rows = [
        "| Model | Accuracy TB(std) | Macro-F1 TB(std) | Recall la_sau TB(std) | Precision la_sau TB(std) | TL score TB(std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, metrics in summary.items():
        rows.append(
            "| {model} | {acc_m:.4f} ({acc_s:.4f}) | {mf1_m:.4f} ({mf1_s:.4f}) | {rec_m:.4f} ({rec_s:.4f}) | {pre_m:.4f} ({pre_s:.4f}) | {tl_m:.4f} ({tl_s:.4f}) |".format(
                model=model,
                acc_m=metrics["accuracy"]["mean"],
                acc_s=metrics["accuracy"]["std"],
                mf1_m=metrics["macro_f1"]["mean"],
                mf1_s=metrics["macro_f1"]["std"],
                rec_m=metrics["recall_la_sau"]["mean"],
                rec_s=metrics["recall_la_sau"]["std"],
                pre_m=metrics["precision_la_sau"]["mean"],
                pre_s=metrics["precision_la_sau"]["std"],
                tl_m=metrics["tl_score"]["mean"],
                tl_s=metrics["tl_score"]["std"],
            )
        )
    return "\n".join(rows) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy multi-seed và tổng hợp TB(std) cho mô hình TL")
    parser.add_argument("--models", type=str, default="vgg16_cbam")
    parser.add_argument("--seeds", type=str, default="42,123,2024")
    parser.add_argument("--min-recall-la-sau", type=float, default=0.75)
    parser.add_argument("--skip-train", action="store_true", help="Bỏ qua train, chỉ evaluate từ weights hiện có")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(REPORTS_DIR / "multi_seed_tl_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=str(REPORTS_DIR / "MULTI_SEED_TL_SUMMARY.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    result = run_experiments(
        models=models,
        seeds=seeds,
        min_recall_la_sau=args.min_recall_la_sau,
        skip_train=args.skip_train,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_content = "# Multi-seed TL summary\n\n"
    md_content += f"Seeds: {result['seeds']}\n\n"
    md_content += f"TL score formula: `{result['tl_score_formula']}`\n\n"
    md_content += _format_table(result["summary_mean_std"])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[SAVED] {out_json}")
    print(f"[SAVED] {out_md}")


if __name__ == "__main__":
    main()
