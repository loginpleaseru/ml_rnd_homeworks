import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

def load_config(path):
    path = Path(path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if cfg["classes"] != CLASSES:
        raise ValueError("Expected the original five-class label mapping")
    for key in ["data_csv", "image_dir", "output_dir"]:
        p = Path(cfg[key]).expanduser()
        cfg[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort repeatability; kernels/hardware can still affect results.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(cfg):
    value = cfg.get("device", "auto")
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value)

def prepare_frame(cfg):
    frame = pd.read_csv(cfg["data_csv"])
    if not {"image_id", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain image_id and label")
    if frame[["image_id", "label"]].isna().any().any() or frame.image_id.duplicated().any():
        raise ValueError("Missing values or duplicate image IDs")
    if not frame.label.isin(range(len(CLASSES))).all():
        raise ValueError("Labels must be integers from 0 to 4")
    if frame.label.nunique() != len(CLASSES):
        raise ValueError("All five classes are required")
    missing = [x for x in frame.image_id if not (Path(cfg["image_dir"]) / x).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, e.g. {missing[:3]}")
    count = cfg.get("sample_size")
    if count and count < len(frame):
        frame, _ = train_test_split(frame, train_size=count, stratify=frame.label, random_state=cfg["seed"])
    elif count and count > len(frame):
        raise ValueError("sample_size exceeds available images")
    frame = frame.reset_index(drop=True)
    if frame.label.value_counts().min() < cfg["n_splits"]:
        raise ValueError("Too few samples per class for n_splits")
    return frame

def make_splits(frame, cfg):
    return StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]).split(frame, frame.label)

def build_model(cfg, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    freeze_most(model, cfg["frozen_ratio"])
    return model

def make_loader(frame, cfg, shuffle=False):
    return DataLoader(CassavaDS(frame, cfg), batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], pin_memory=get_device(cfg).type == "cuda")

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

class CassavaDS(Dataset):
    def __init__(self, frame, cfg):
        self.cfg = cfg
        self.frame = frame.reset_index(drop=True)
        self.tf = T.Compose([T.Resize((self.cfg["size"], self.cfg["size"])), T.ToTensor(), T.Normalize(self.cfg["mean"], self.cfg["std"])])
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        x = self.tf(Image.open(Path(self.cfg["image_dir"]) / row["image_id"]).convert("RGB"))
        return x, int(row["label"])

def freeze_most(model, frozen_ratio=0.7):
    params = list(model.parameters())
    for i, p in enumerate(params):
        p.requires_grad = i >= int(frozen_ratio * len(params))

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
def evaluate(model, loader, criterion, device):
    model.eval()
    ys, probs = [], []
    total_loss, total_samples = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
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
