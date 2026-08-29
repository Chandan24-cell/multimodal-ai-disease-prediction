"""
Production Training Script for NIH Chest X-ray14 Dataset
Trains on all 112,120 images for medical-grade performance
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
writer = SummaryWriter("runs/training")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]


class NIHChestXRayDataset(Dataset):
    """NIH Chest X-ray14 Dataset for multi-label classification."""

    def __init__(self, csv_file, image_dirs, transform=None, sample_frac=1.0):
        self.image_dirs = image_dirs
        self.transform = transform

        logger.info("Loading metadata from %s...", csv_file)
        self.data = pd.read_csv(csv_file)

        if sample_frac < 1.0:
            self.data = self.data.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            logger.info("Sampled %s images (%.1f%% of dataset)", len(self.data), sample_frac * 100)

        logger.info("Building image path lookup...")
        self.image_paths = {}
        for image_dir in image_dirs:
            if os.path.exists(image_dir):
                for image_name in os.listdir(image_dir):
                    self.image_paths[image_name] = os.path.join(image_dir, image_name)

        logger.info("Found %s images in directories", len(self.image_paths))
        logger.info("Dataset size: %s samples", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_name = row["Image Index"]
        labels_string = str(row["Finding Labels"])

        image_path = self.image_paths.get(image_name)
        if not image_path or not os.path.exists(image_path):
            image = Image.new("RGB", (224, 224), color="black")
            logger.warning("Image not found: %s", image_name)
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as error:
                image = Image.new("RGB", (224, 224), color="black")
                logger.warning("Failed to load image %s: %s", image_name, error)

        if self.transform:
            image = self.transform(image)

        label_vector = [0.0] * len(DISEASE_LABELS)
        if labels_string not in ("No Finding", "nan") and labels_string.strip():
            for finding in labels_string.split("|"):
                finding = finding.strip()
                if finding in DISEASE_LABELS:
                    label_vector[DISEASE_LABELS.index(finding)] = 1.0

        return {
            "pixel_values": image,
            "labels": torch.tensor(label_vector, dtype=torch.float32)
        }


def _resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_medical_vit_production(
    csv_file="datasets/chest_xray/nih_extracted/Data_Entry_2017.csv",
    image_dirs=None,
    epochs=10,
    batch_size=32,
    lr=2e-4,
    output_dir="backend/models/vit/medical_finetuned",
    sample_frac=1.0
):
    """Train ViT on the full NIH Chest X-ray14 dataset."""
    logger.info("=" * 80)
    logger.info("NIH CHEST X-RAY14 PRODUCTION TRAINING")
    logger.info("=" * 80)

    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    csv_path = _resolve_path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    # Default image directories
    if image_dirs is None:
        base_dir = 'datasets/chest_xray/nih_extracted'
        # Notice the extra 'images' folder at the end of the path!
        image_dirs = [
            os.path.join(base_dir, f'images_{i:03d}', 'images')
            for i in range(1, 13)  # images_001 to images_012
        ]

    image_dirs = [str(_resolve_path(image_dir)) for image_dir in image_dirs]
    existing_dirs = [image_dir for image_dir in image_dirs if os.path.exists(image_dir)]
    if not existing_dirs:
        raise FileNotFoundError("No image directories found. Extract the dataset first.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = NIHChestXRayDataset(csv_path, existing_dirs, transform, sample_frac)
    if len(dataset) < 2:
        raise ValueError("At least two dataset samples are required for training and validation.")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(DISEASE_LABELS),
        ignore_mismatched_sizes=True
    ).to(device)

    for name, parameter in model.named_parameters():
        if "classifier" not in name:
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val_loss = float("inf")
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            global_step = epoch * len(train_loader) + batch_index
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_index % 100 == 0:
                logger.info(
                    "Epoch %s/%s, Batch %s/%s, Loss: %.4f, Time: %.1fs",
                    epoch + 1, epochs, batch_index, len(train_loader), loss.item(), time.time() - epoch_start
                )

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values)
                val_loss += criterion(outputs.logits, labels).item()
                val_preds.extend(torch.sigmoid(outputs.logits).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.asarray(val_preds)
        val_labels = np.asarray(val_labels)
        try:
            val_auc = roc_auc_score(val_labels, val_preds, average="macro")
        except ValueError:
            val_auc = 0.0
        val_f1 = f1_score(val_labels, (val_preds >= 0.5).astype(int), average="macro", zero_division=0)
        scheduler.step(avg_val_loss)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        writer.add_scalar("validation/auc", val_auc, epoch)
        writer.add_scalar("validation/f1", val_f1, epoch)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.flush()

        logger.info(
            "Epoch %s/%s: train_loss=%.4f val_loss=%.4f val_auc=%.4f val_f1=%.4f time=%.1fs",
            epoch + 1, epochs, avg_train_loss, avg_val_loss, val_auc, val_f1, time.time() - epoch_start
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_path)
            logger.info("Saved best model to %s", output_path)

    writer.close()
    logger.info("Training complete in %.2f hours", (time.time() - start_time) / 3600)
    return model


if __name__ == "__main__":
    train_medical_vit_production(epochs=10, batch_size=16, lr=2e-4, sample_frac=1.0)
