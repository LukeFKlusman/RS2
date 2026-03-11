"""
train_letter_cnn.py
════════════════════════════════════════════════════════════
Train the LetterCNN on images collected by collect_training_data.py

USAGE:
  python3 train_letter_cnn.py

EXPECTS:
  data/raw/
    A/  *.png
    B/  *.png
    ...

OUTPUTS:
  letter_cnn.pt          ← weights file (use in realsense_camera_cnn.py)
  training_report.txt    ← per-class accuracy breakdown

REQUIREMENTS:
  pip install torch torchvision scikit-learn matplotlib
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ── Hyperparameters ────────────────────────────────────────
DATA_ROOT      = "data/raw"
OUTPUTS_DIR    = "outputs"          # all graphs, reports, and model saved here
MODEL_OUT      = os.path.join(OUTPUTS_DIR, "letter_cnn.pt")
IMG_SIZE       = 64
BATCH_SIZE     = 64
EPOCHS         = 25
LR             = 1e-3
VAL_SPLIT      = 0.15
TEST_SPLIT     = 0.10
LABEL_MAP      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#LABEL_MAP      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
NUM_CLASSES    = len(LABEL_MAP)
# ──────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class LetterDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []   # (path, label_idx)
        self.transform = transform
        missing        = []

        for idx, label in enumerate(LABEL_MAP):
            folder = os.path.join(root, label)
            if not os.path.exists(folder):
                missing.append(label)
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
            if not files:
                missing.append(label)
            for f in files:
                self.samples.append((os.path.join(folder, f), idx))

        if missing:
            print(f"[WARNING] Missing or empty classes: {missing}")
            print("  Collect more data for these letters before training.")

        print(f"[Dataset] {len(self.samples)} images across "
              f"{NUM_CLASSES - len(missing)}/{NUM_CLASSES} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')  # greyscale
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════
# MODEL  (must match realsense_camera_cnn.py exactly)
# ══════════════════════════════════════════════════════════

class LetterCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32×32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 16×16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"[Train] Outputs will be saved to: {OUTPUTS_DIR}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Augmentation for training ─────────────────────────
    # Key augmentations that fix your B/8, O/0, I/1 problems:
    # RandomRotation handles tilted letters on camera
    # ColorJitter handles different lighting conditions
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # simulates angled camera view
        transforms.RandomHorizontalFlip(p=0.0),    # DO NOT flip letters!
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # ── Load and split dataset ────────────────────────────
    full_dataset = LetterDataset(DATA_ROOT, transform=train_tf)
    if len(full_dataset) == 0:
        print("[ERROR] No training images found. Run collect_training_data.py first.")
        sys.exit(1)

    n_total = len(full_dataset)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = int(n_total * TEST_SPLIT)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    # Validation/test use clean transforms
    val_ds.dataset  = LetterDataset(DATA_ROOT, transform=val_tf)
    test_ds.dataset = LetterDataset(DATA_ROOT, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[Train] Split → train: {n_train}  val: {n_val}  test: {n_test}")

    # ── Model, loss, optimiser ────────────────────────────
    model     = LetterCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    print(f"\n[Train] Starting {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        # ── Training pass ─────────────────────────────────
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        avg_loss = running_loss / n_train

        # ── Validation pass ───────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / n_val * 100

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss: {avg_loss:.4f}  "
              f"val_acc: {val_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"    ✓ Saved best model → {MODEL_OUT}")

        # ── Periodic Confusion Matrix (Every 5 Epochs) ────────
        if epoch % 5 == 0:
                    model.eval()
                    temp_preds, temp_labels = [], []
                    with torch.no_grad():
                        for imgs, labels in val_loader:
                            imgs = imgs.to(device)
                            preds = model(imgs).argmax(dim=1).cpu().numpy()
                            temp_preds.extend(preds)
                            temp_labels.extend(labels.numpy())
                    
                    # Generate and save the matrix for this checkpoint
                    cm = confusion_matrix(temp_labels, temp_preds)
                    plt.figure(figsize=(16, 14))
                    
                    # This handles the Actual (Y) vs Predicted (X) labeling
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                                xticklabels=LABEL_MAP, yticklabels=LABEL_MAP)
                    
                    plt.title(f"Confusion Matrix - Epoch {epoch} (Val Acc: {val_acc:.1f}%)")
                    plt.ylabel("ACTUAL LETTER (True Label)", fontsize=12, fontweight='bold')
                    plt.xlabel("PREDICTED LETTER (Model Guess)", fontsize=12, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUTS_DIR, f"confusion_matrix_epoch_{epoch}.png"))
                    plt.close()
                    print(f"    ✓ Intermediate matrix saved → epoch_{epoch}.png")
    # ── Test evaluation ───────────────────────────────────
    print(f"\n[Eval] Loading best model for test set evaluation...")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_MAP, zero_division=0
    )
    print("\n" + report)

    # Save text report
    with open(os.path.join(OUTPUTS_DIR, "training_report.txt"), "w") as f:
        f.write(f"Best val accuracy: {best_val_acc:.1f}%\n\n")
        f.write(report)
    print("[Done] Report saved to outputs/training_report.txt")

    # ── Confusion matrix plot ─────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_MAP, yticklabels=LABEL_MAP)
    plt.title(f"Confusion Matrix (Test Set) — Best val acc: {best_val_acc:.1f}%")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    print("[Done] Confusion matrix saved to outputs/confusion_matrix.png")

    # ── Loss / accuracy curve ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses);  ax1.set_title("Training Loss");    ax1.set_xlabel("Epoch")
    ax2.plot(val_accs);      ax2.set_title("Validation Acc %"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "training_curves.png"), dpi=150)
    print("[Done] Training curves saved to outputs/training_curves.png")
    print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"   Model saved to: {MODEL_OUT}")


if __name__ == '__main__':
    train()