"""Run the four fixed-split experiments from the original notebook."""
import argparse
import json
from pathlib import Path
import utils as u
from utils import train_split, confusion_matrix, np, pd, CLASSES, EXPS, ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    cfg = u.load_config(args.config)
    u.configure(cfg)
    manifest, tr_df, va_df = u.load_split(cfg)
    print(f"train={len(tr_df)}, val={len(va_df)}, device={u.DEVICE}")
    if args.check_data:
        return
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output_dir; refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "subset_and_split.csv", index=False)
    runtime = dict(cfg, torch=str(u.torch.__version__), torchvision=str(u.torchvision.__version__),
                   numpy=u.np.__version__, pandas=u.pd.__version__, sklearn=u.sklearn.__version__,
                   python=u.platform.python_version(), actual_device=str(u.DEVICE))
    (out / "config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2))
    selected = EXPS
    summary_rows = []
    histories, per_class, cms, val_predictions = {}, {}, {}, {}
    for name, hypothesis, cw, aug in selected:
        exp_dir = out / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        result = train_split(
            tr_df, va_df, run_seed=cfg["training_seed"],
            class_weights_flag=cw, augment=aug, desc=name,
            ckpt_path=exp_dir / "best.pt",
        )
        histories[name] = result["history"]
        result["history"].to_csv(exp_dir / "history.csv", index=False)
        yt, yp, prob = result["y_true"], result["y_pred"], result["prob"]
        assert np.array_equal(yt, va_df["label"].to_numpy())
        pc = result["per_class"]
        healthy_mask = yt == CLASSES.index("Healthy")
        healthy_error_rate = (yp[healthy_mask] != CLASSES.index("Healthy")).sum() / healthy_mask.sum()
        summary_rows.append({
            "name": name, "hypothesis": hypothesis, "seed": cfg["training_seed"],
            "class_weights": cw, "augment": aug, "best_epoch": result["best_epoch"],
            **{k: result[k] for k in ["loss", "f1_macro", "precision_macro", "recall_macro", "pr_auc_macro"]},
            "f1_healthy": pc.loc["Healthy", "f1"],
            "healthy_to_disease_rate": healthy_error_rate,
            "ckpt_path": result["ckpt_path"],
        })
        per_class[name], cms[name] = pc, confusion_matrix(yt, yp, labels=range(len(CLASSES)))
        predictions = va_df.copy()
        predictions["prediction"] = yp
        for k, label in enumerate(CLASSES):
            predictions[f"prob_{label}"] = prob[:, k]
        predictions.to_csv(exp_dir / "val_predictions.csv", index=False)
        pc.to_csv(exp_dir / "val_per_class.csv")
        val_predictions[name] = predictions
        # Сохраняем сводку после каждого завершённого эксперимента.
        log = pd.DataFrame(summary_rows)
        base_f1 = log.loc[log["name"] == "baseline", "f1_macro"].iloc[0]
        log["delta_f1"] = log["f1_macro"] - base_f1
        log.to_csv(out / "experiments_hw2.csv", index=False)
        print(f"{name}: метрики по классам на validation")
        print(pc)

    print("Метрики на общем validation и разность macro F1 с baseline:")
    print(log)

if __name__ == "__main__":
    main()
