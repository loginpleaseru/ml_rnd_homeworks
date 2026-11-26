"""Verify saved validation predictions and rebuild tables, plots and error examples."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import utils as u
from utils import CLASSES, EXPS, ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    parser.add_argument("--results", default=str(ROOT / "reports"))
    parser.add_argument("--checkpoints", help="Optional folder of <name>.pt to verify inference")
    parser.add_argument("--predict", action="store_true", help="Evaluate checkpoints from a newly trained run")
    args = parser.parse_args()
    cfg = u.load_config(args.config)
    u.configure(cfg)
    _, _, va = u.load_split(cfg, check_images=bool(args.checkpoints or args.predict))
    root = Path(args.results).resolve()
    archived = (root / "experiments").is_dir()
    exp_root = root / "experiments" if archived else root
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(root / "experiments_hw2.csv")
    if set(log.name) != {e[0] for e in EXPS}:
        raise ValueError("All four experiment rows are required")
    cms, predictions, rows = {}, {}, []
    for name, _, _, _ in EXPS:
        pred = pd.read_csv(exp_root / name / "val_predictions.csv")
        if not pred[["image_id", "label"]].equals(va[["image_id", "label"]]):
            raise ValueError(f"Validation object order mismatch: {name}")
        prob = pred[[f"prob_{c}" for c in CLASSES]].to_numpy()
        if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(1), 1, atol=1e-5):
            raise ValueError(f"Invalid probabilities: {name}")
        if not np.array_equal(prob.argmax(1), pred.prediction):
            raise ValueError(f"Labels disagree with argmax: {name}")
        if args.checkpoints or args.predict:
            checkpoint_path = Path(args.checkpoints) / f"{name}.pt" if args.checkpoints else exp_root / name / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(u.DEVICE)
            loader = DataLoader(u.CassavaDS(va), batch_size=cfg["batch"], shuffle=False, num_workers=cfg["num_workers"])
            _, yt, yp, actual = u.evaluate(model, loader, nn.CrossEntropyLoss())
            if checkpoint["epoch"] != int(log.set_index("name").loc[name, "best_epoch"]):
                raise ValueError("Checkpoint epoch mismatch")
            if args.predict:
                pred["prediction"] = yp
                pred[[f"prob_{c}" for c in CLASSES]] = actual
                prob = actual
                pred.to_csv(exp_root / name / "val_predictions.csv", index=False)
            elif not np.allclose(actual, prob, atol=2e-4, rtol=2e-3):
                raise ValueError(f"Checkpoint inference differs from saved predictions: {name}")
        yt, yp = pred.label.to_numpy(), pred.prediction.to_numpy()
        metrics = u.classification_metrics(yt, yp, prob)
        pc = u.per_class_table(yt, yp)
        if not np.allclose(pc.to_numpy(), pd.read_csv(exp_root / name / "val_per_class.csv", index_col=0).to_numpy(), atol=1e-7):
            raise ValueError(f"Per-class table mismatch: {name}")
        healthy = yt == 4
        metrics.update(f1_healthy=pc.loc["Healthy", "f1"], healthy_to_disease_rate=float((yp[healthy] != 4).mean()))
        saved = log.set_index("name").loc[name]
        for key, value in metrics.items():
            if not np.isclose(value, saved[key], atol=1e-6):
                raise ValueError(f"Metric mismatch: {name}/{key}")
        history = pd.read_csv(exp_root / name / "history.csv")
        if len(history) != cfg["epochs"] or history.epoch.tolist() != list(range(1, cfg["epochs"]+1)):
            raise ValueError(f"Incomplete history: {name}")
        best = history.loc[history.val_f1_macro.idxmax()]
        if int(best.epoch) != saved.best_epoch or not np.isclose(best.val_f1_macro, metrics["f1_macro"]):
            raise ValueError(f"Best epoch mismatch: {name}")
        cm = confusion_matrix(yt, yp, labels=range(5))
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(exp_root / name / "confusion_matrix.csv")
        errors = pred[pred.label != pred.prediction]
        errors.to_csv(exp_root / name / "errors.csv", index=False)
        cms[name], predictions[name] = cm, pred
        rows.append(dict(name=name, **metrics))
    base_f1 = log.set_index("name").loc["baseline", "f1_macro"]
    if not np.allclose(log.delta_f1, log.f1_macro - base_f1):
        raise ValueError("delta_f1 mismatch")
    pd.DataFrame(rows).to_csv(root / "verified_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, _, _, _) in zip(axes, EXPS):
        ax.imshow(cms[name], cmap="Blues")
        ax.set_xticks(range(5), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(5), CLASSES)
        ax.set_title(name)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrices.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for name, _, _, _ in EXPS:
        h = pd.read_csv(exp_root / name / "history.csv")
        for ax, metric in zip(axes.ravel(), ["train_loss", "val_loss", "val_f1_macro", "val_pr_auc_macro"]):
            ax.plot(h.epoch, h[metric], label=name)
            ax.set_title(metric); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    for ax in axes.ravel(): ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "training_curves.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["f1_macro", "f1_healthy", "healthy_to_disease_rate"]):
        log.set_index("name")[metric].plot.bar(ax=ax)
        ax.set_title(metric); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figures / "val_comparison.png", dpi=150); plt.close(fig)
    # Pick two complementary Healthy examples: corrected and persistent errors.
    base = predictions["baseline"].set_index("image_id")
    full = predictions["full_h1_h2"].set_index("image_id")
    candidates = base[(base.label == 4) & (base.prediction != 4)]
    fixed = [i for i in candidates.index if full.loc[i, "prediction"] == 4]
    persistent = [i for i in candidates.index if full.loc[i, "prediction"] != 4]
    ids = fixed[:1] + persistent[:1]
    records = [dict(image_id=i, label=4, baseline=int(base.loc[i,"prediction"]), full_h1_h2=int(full.loc[i,"prediction"])) for i in ids]
    pd.DataFrame(records).to_csv(root / "error_examples.csv", index=False)
    if ids and all((Path(cfg["image_dir"]) / i).is_file() for i in ids):
        fig, axes = plt.subplots(1, len(ids), figsize=(5*len(ids), 4), squeeze=False)
        for ax, i in zip(axes.ravel(), ids):
            with Image.open(Path(cfg["image_dir"]) / i) as im: ax.imshow(im.convert("RGB"))
            ax.set_title(f"{i}; true: Healthy\nbaseline: {CLASSES[int(base.loc[i,'prediction'])]} → full: {CLASSES[int(full.loc[i,'prediction'])]}")
            ax.axis("off")
        fig.tight_layout(); fig.savefig(figures / "error_examples.png", dpi=150); plt.close(fig)
    (root / "verification.json").write_text(json.dumps(dict(experiments=4, validation_size=len(va),
        metrics_match=True, split_match=True, histories_match=True,
        checkpoint_inference_checked=bool(args.checkpoints or args.predict)), indent=2))
    print("OK: four experiments, common validation, metrics, per-class tables and best epochs")

if __name__ == "__main__":
    main()
