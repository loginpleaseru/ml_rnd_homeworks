"""Train the notebook baseline with three stratified folds."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
from utils import (ROOT, CLASSES, load_config, seed_all, get_device, prepare_frame,
                   make_splits, build_model, make_loader, evaluate, save_json)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    parser.add_argument("--check-data", action="store_true", help="Validate data/splits without training")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    frame = prepare_frame(cfg)
    splits = list(make_splits(frame, cfg))
    print(f"Samples: {len(frame)}; class counts: {frame.label.value_counts().sort_index().to_dict()}")
    print(f"Fold sizes: {[(len(t), len(v)) for t, v in splits]}")
    if args.check_data:
        return
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new output_dir.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    assignment = np.zeros(len(frame), dtype=int)
    for fold, (_, va) in enumerate(splits, 1):
        assignment[va] = fold
    frame.assign(fold=assignment).to_csv(out / "splits.csv", index=False)
    device = get_device(cfg)
    rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        train_loader = make_loader(frame.iloc[tr], cfg, shuffle=True)
        val_loader = make_loader(frame.iloc[va], cfg)
        model = build_model(cfg).to(device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        best_f1, best_state, best_epoch = -1., None, None
        history = []
        checkpoint = out / "checkpoints" / f"resnet18_fold_{fold}_best.pt"
        for ep in tqdm(range(cfg["epochs"]), desc=f"fold {fold}"):
            model.train()
            total_loss, total, ys, preds = 0., 0, [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                total += len(y)
                ys.extend(y.detach().cpu().tolist())
                preds.extend(logits.detach().argmax(1).cpu().tolist())
            f1 = precision_recall_fscore_support(ys, preds, labels=range(5), average="macro", zero_division=0)[2]
            metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
            history.append(dict(epoch=ep+1, train_loss=total_loss/total, train_f1_macro=f1,
                                **{f"val_{k}": v for k, v in metrics.items()}))
            if metrics["f1_macro"] > best_f1:
                best_f1, best_epoch = metrics["f1_macro"], ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(dict(state_dict=best_state, fold=fold, epoch=best_epoch,
                                val_f1_macro=best_f1, config=cfg), checkpoint)
            pd.DataFrame(history).to_csv(out / f"fold_{fold}_history.csv", index=False)
        model.load_state_dict(best_state)
        metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        rows.append(dict(fold=fold, best_epoch=best_epoch, **metrics, ckpt_path=str(checkpoint)))
        pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    summary = pd.DataFrame(rows)[["f1_macro", "precision_macro", "recall_macro", "pr_auc_macro"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    print(summary)

if __name__ == "__main__":
    main()
