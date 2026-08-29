import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """NIH ChestX-ray14 or CheXpert dataset loader."""

    def __init__(self, image_dir, labels_file, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size

        # Load labels
        self.data = pd.read_csv(labels_file)

        # Define disease labels (NIH ChestX-ray14)
        self.disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        # Load and convert image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels (multi-label classification)
        labels = self.data.iloc[idx, 1:15].values.astype(float)

        return {
            "pixel_values": image,
            "labels": torch.tensor(labels, dtype=torch.float32)
        }


def train_medical_vit(
    train_dataset,
    val_dataset,
    num_epochs=10,
    batch_size=32,
    learning_rate=2e-5,
    output_dir="models/vit/medical_finetuned"
):
    """Fine-tune ViT on medical imaging data."""
    logger.info("Starting medical ViT fine-tuning...")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load pre-trained ViT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=14,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values)
                val_loss += criterion(outputs.logits, labels).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            logger.info(f"  Saved best model to {output_dir}")

    logger.info("Training complete!")
    return model


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ChestXRayDataset(
        image_dir="datasets/chest_xray/nih/images",
        labels_file="datasets/chest_xray/nih/Data_Entry_2017.csv",
        transform=transform
    )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    model = train_medical_vit(train_dataset, val_dataset)
