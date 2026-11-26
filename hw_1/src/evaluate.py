"""Evaluate all saved fold models on their held-out images and export OOF errors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from utils import (ROOT, CLASSES, load_config, get_device, build_model, make_loader,
                   evaluate, classification_metrics, per_class_table, save_json)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    # The run config is authoritative; data paths may be overridden for relocation.
    run_cfg = load_config(out / "config.yaml")
    run_cfg["image_dir"] = cfg["image_dir"]
    cfg = run_cfg
    frame = pd.read_csv(out / "splits.csv")
    if frame.image_id.duplicated().any() or not frame.fold.isin(range(1, cfg["n_splits"]+1)).all():
        raise ValueError("Invalid saved fold assignments")
    device = get_device(cfg)
    probs = np.full((len(frame), len(CLASSES)), np.nan)
    for fold in range(1, cfg["n_splits"]+1):
        idx = np.flatnonzero(frame.fold.to_numpy() == fold)
        path = out / "checkpoints" / f"resnet18_fold_{fold}_best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["fold"] != fold:
            raise ValueError("Checkpoint fold mismatch")
        model = build_model(cfg, pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        _, _, _, prob = evaluate(model, make_loader(frame.iloc[idx], cfg), nn.CrossEntropyLoss(), device)
        probs[idx] = prob
    if not np.isfinite(probs).all():
        raise ValueError("Incomplete OOF predictions")
    true, pred = frame.label.to_numpy(), probs.argmax(1)
    result = frame.assign(predicted_label=pred)
    for i, name in enumerate(CLASSES):
        result[f"prob_{name}"] = probs[:, i]
    result.to_csv(out / "oof_predictions.csv", index=False)
    save_json(out / "oof_metrics.json", classification_metrics(true, pred, probs))
    per_class_table(true, pred).to_csv(out / "per_class.csv")
    pd.DataFrame(confusion_matrix(true, pred, labels=range(5)), index=CLASSES, columns=CLASSES).to_csv(out / "confusion_matrix.csv")
    errors = result[result.label != result.predicted_label]
    errors.to_csv(out / "errors.csv", index=False)
    healthy = errors[errors.label == 4]
    examples = pd.concat([healthy, errors[errors.label != 4]]).head(3)
    if len(examples):
        fig, axes = plt.subplots(1, len(examples), figsize=(5*len(examples), 4), squeeze=False)
        for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
            with Image.open(Path(cfg["image_dir"]) / row.image_id) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{row.image_id}\n{CLASSES[row.label]} → {CLASSES[row.predicted_label]} (fold {row.fold})")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "error_examples.png", dpi=150)
        plt.close(fig)
    print(classification_metrics(true, pred, probs))

if __name__ == "__main__":
    main()
