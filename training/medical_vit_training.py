"""
Medical ViT Training Script for NIH Chest X-ray14 Dataset
Multi-label classification with 14 disease categories
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]


class NIHChestXRayDataset(Dataset):
    """
    NIH Chest X-ray14 Dataset for multi-label classification.

    Input: CSV file with Image Index and Finding Labels.
    Output: Multi-hot encoded labels with 14 dimensions.
    """

    def __init__(self, csv_file, image_dirs, image_list_file=None, transform=None, max_samples=None):
        self.image_dirs = image_dirs
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        if image_list_file and os.path.exists(image_list_file):
            with open(image_list_file, "r", encoding="utf-8") as file:
                image_names = [line.strip() for line in file]
            self.data = self.data[self.data["Image Index"].isin(image_names)]
            logger.info("Loaded %s images from official split", len(self.data))

        if max_samples:
            self.data = self.data.head(max_samples)
            logger.info("Limited dataset to %s samples", max_samples)

        logger.info("Total samples: %s", len(self.data))

        self.image_paths = {}
        for image_dir in image_dirs:
            if os.path.exists(image_dir):
                for image_name in os.listdir(image_dir):
                    self.image_paths[image_name] = os.path.join(image_dir, image_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_name = row["Image Index"]
        labels_string = str(row["Finding Labels"])

        image_path = self.image_paths.get(image_name)
        if not image_path:
            image = Image.new("RGB", (224, 224), color="black")
            logger.warning("Image not found: %s", image_name)
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_vector = [0.0] * len(DISEASE_LABELS)
        if labels_string not in ("No Finding", "nan"):
            for finding in labels_string.split("|"):
                finding = finding.strip()
                if finding in DISEASE_LABELS:
                    label_vector[DISEASE_LABELS.index(finding)] = 1.0

        return {
            "pixel_values": image,
            "labels": torch.tensor(label_vector, dtype=torch.float32)
        }


def train_medical_vit(
    csv_file="datasets/chest_xray/nih/Data_Entry_2017.csv",
    image_dirs=None,
    train_list="datasets/chest_xray/nih/train_val_list.txt",
    val_list="datasets/chest_xray/nih/test_list.txt",
    max_samples=5000,
    epochs=5,
    batch_size=16,
    lr=2e-4,
    output_dir="backend/models/vit/medical_finetuned"
):
    """Train ViT on the NIH Chest X-ray14 dataset."""
    logger.info("=" * 60)
    logger.info("Starting Medical ViT Training on NIH Chest X-ray14")
    logger.info("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if image_dirs is None:
        image_dirs = [
            f"datasets/chest_xray/nih/images_{index:03d}"
            for index in range(1, 13)
        ]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    logger.info("Loading training dataset...")
    train_dataset = NIHChestXRayDataset(
        csv_file=csv_file,
        image_dirs=image_dirs,
        image_list_file=train_list,
        transform=transform,
        max_samples=max_samples
    )

    logger.info("Loading validation dataset...")
    val_dataset = NIHChestXRayDataset(
        csv_file=csv_file,
        image_dirs=image_dirs,
        image_list_file=val_list,
        transform=transform,
        max_samples=int(max_samples * 0.2)
    )

    if not train_dataset or not val_dataset:
        raise ValueError("Training and validation datasets must each contain at least one sample.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info("Loading pre-trained ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(DISEASE_LABELS),
        ignore_mismatched_sizes=True
    )
    model.to(device)

    for name, parameter in model.named_parameters():
        if "classifier" not in name:
            parameter.requires_grad = False
    logger.info("Frozen ViT base layers. Training classifier head only.")

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_index, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs.logits) >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()

            if batch_index % 50 == 0:
                logger.info(
                    "Epoch %s/%s, Batch %s/%s, Loss: %.4f",
                    epoch + 1, epochs, batch_index, len(train_loader), loss.item()
                )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values)
                val_loss += criterion(outputs.logits, labels).item()
                predictions = (torch.sigmoid(outputs.logits) >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.numel()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        logger.info("Epoch %s/%s Summary:", epoch + 1, epochs)
        logger.info("  Train Loss: %.4f, Train Acc: %.4f", avg_train_loss, train_accuracy)
        logger.info("  Val Loss: %.4f, Val Acc: %.4f", avg_val_loss, val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            logger.info("Saved best model to %s (Val Loss: %.4f)", output_dir, best_val_loss)

    logger.info("Medical ViT Training Complete!")
    logger.info("Best Validation Loss: %.4f", best_val_loss)
    logger.info("Model saved to: %s", output_dir)
    return model


if __name__ == "__main__":
    train_medical_vit(
        max_samples=5000,
        epochs=3,
        batch_size=16,
        lr=2e-4
    )
