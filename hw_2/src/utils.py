import os
import random
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import sklearn
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm


import yaml
ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
EXPS = [
    ("baseline", "Без весов классов и аугментаций", False, False),
    ("only_h1", "H1: взвешенный CrossEntropy", True, False),
    ("only_h2", "H2: аугментации", False, True),
    ("full_h1_h2", "H1+H2", True, True),
]

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ["manifest", "image_dir", "output_dir"]:
        p = Path(cfg[key]).expanduser()
        cfg[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg

def configure(cfg):
    global SEED, SIZE, BATCH, EPOCHS, LR, WEIGHT_DECAY, FREEZE_RATIO
    global NUM_WORKERS, MEAN, STD, TRAIN_PATH, DEVICE
    SEED, SIZE, BATCH, EPOCHS, LR = (cfg[k] for k in ["seed", "size", "batch", "epochs", "lr"])
    WEIGHT_DECAY, FREEZE_RATIO = cfg["weight_decay"], cfg["freeze_ratio"]
    NUM_WORKERS, MEAN, STD = cfg["num_workers"], cfg["mean"], cfg["std"]
    TRAIN_PATH = Path(cfg["image_dir"])
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device)

def load_split(cfg, check_images=True):
    path = Path(cfg["manifest"])
    if cfg.get("manifest_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != cfg["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    manifest = pd.read_csv(path)
    if manifest[["image_id", "label", "split"]].isna().any().any() or not manifest.image_id.is_unique:
        raise ValueError("Missing values or repeated image IDs")
    if set(manifest.split) != {"train", "val"} or set(manifest.label) != set(range(5)):
        raise ValueError("Invalid split or class labels")
    if len(manifest) != cfg["subset_size"]:
        raise ValueError("subset_size differs from saved manifest")
    # Recover the original row order, including the shuffle performed by train_test_split.
    frame = manifest[["image_id", "label"]].copy()
    tr, va = train_test_split(frame, test_size=cfg["val_size"], stratify=frame.label, random_state=cfg["seed"])
    if set(tr.image_id) != set(manifest.loc[manifest.split == "train", "image_id"]):
        raise ValueError("Config does not reproduce the saved train/val split")
    if check_images:
        missing = [x for x in manifest.image_id if not (Path(cfg["image_dir"]) / x).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images: {missing[:3]}")
    return manifest, tr.reset_index(drop=True), va.reset_index(drop=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CassavaDS(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        base = [T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)]
        aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.05)] if augment else []
        self.tf = T.Compose(aug + base)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        x = self.tf(Image.open(TRAIN_PATH / row["image_id"]).convert("RGB"))
        return x, int(row["label"])

def freeze_most(model, ratio=0.7):
    params = list(model.parameters())
    n = len(params)
    for i, p in enumerate(params):
        p.requires_grad = i >= int(ratio * n)

def class_weights(tr):
    c = np.bincount(tr["label"].values, minlength=len(CLASSES)).astype(np.float64)
    assert np.all(c > 0), "В train отсутствует класс"
    w = len(tr) / (len(CLASSES) * c)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def classification_metrics(y_true, y_pred, prob):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), average="macro", zero_division=0
    )
    y_bin = label_binarize(y_true, classes=range(len(CLASSES)))
    return {
        "f1_macro": f1,
        "precision_macro": p,
        "recall_macro": r,
        "pr_auc_macro": average_precision_score(y_bin, prob, average="macro"),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    ys, probs = [], []
    total_loss, total_samples = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        total_samples += y.size(0)
        ys.append(y.cpu().numpy())
        probs.append(logits.softmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    prob = np.vstack(probs)
    y_pred = prob.argmax(1)
    metrics = classification_metrics(y_true, y_pred, prob)
    metrics["loss"] = total_loss / total_samples
    return metrics, y_true, y_pred, prob


def per_class_table(y_true, y_pred):
    p, r, f1, n = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f1, "support": n}, index=CLASSES
    )

def save_ckpt(path, state_dict, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, **meta}, path)


def train_split(tr, va, *, run_seed, class_weights_flag=False, augment=False,
                desc="train", ckpt_path=None):
    seed_everything(run_seed)
    train_generator = torch.Generator().manual_seed(run_seed)
    val_generator = torch.Generator().manual_seed(run_seed + 10000)
    tr_ld = DataLoader(CassavaDS(tr, augment=augment), batch_size=BATCH, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                       worker_init_fn=seed_worker, generator=train_generator)
    va_ld = DataLoader(CassavaDS(va, augment=False), batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                       worker_init_fn=seed_worker, generator=val_generator)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    freeze_most(model, FREEZE_RATIO)
    model = model.to(DEVICE)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = class_weights(tr) if class_weights_flag else None
    crit = nn.CrossEntropyLoss(weight=weights)
    val_crit = nn.CrossEntropyLoss()  # одинаковый validation loss для всех вариантов
    best_f1, best, best_epoch = -1.0, None, None
    history = []
    for ep in tqdm(range(EPOCHS), desc=desc):
        model.train()
        total_loss, total_samples = 0.0, 0
        train_true, train_pred = [], []
        for x, y in tqdm(tr_ld, desc=f"ep {ep + 1}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            # У weighted CE знаменатель — сумма весов целевых классов.
            denominator = y.size(0) if weights is None else weights[y].sum().item()
            total_loss += loss.item() * denominator
            total_samples += denominator
            train_true.extend(y.detach().cpu().tolist())
            train_pred.extend(logits.detach().argmax(1).cpu().tolist())
        _, _, train_f1, _ = precision_recall_fscore_support(
            train_true, train_pred, labels=range(len(CLASSES)), average="macro", zero_division=0
        )
        val_metrics, _, _, _ = evaluate(model, va_ld, val_crit)
        history.append({
            "epoch": ep + 1,
            "train_loss": total_loss / total_samples,
            "train_f1_macro": train_f1,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        f1 = val_metrics["f1_macro"]
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep + 1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                save_ckpt(ckpt_path, best, {"val_f1_macro": best_f1, "epoch": best_epoch})
        tqdm.write(
            f"{desc} ep {ep + 1}: train loss={history[-1]['train_loss']:.4f}, "
            f"val loss={val_metrics['loss']:.4f}, val F1={f1:.4f} (best={best_f1:.4f})"
        )
    model.load_state_dict(best)
    metrics, yt, yp, prob = evaluate(model, va_ld, val_crit)
    return {
        **metrics, "best_epoch": best_epoch,
        "per_class": per_class_table(yt, yp),
        "history": pd.DataFrame(history),
        "y_true": yt, "y_pred": yp, "prob": prob,
        "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
    }
