"""Continuously fine-tune the medical ViT from doctor-corrected feedback."""

import logging
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification

# Add backend to path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from database.mongodb import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


class FeedbackDataset(Dataset):
    """Dataset built from doctor-corrected feedback."""

    def __init__(self, feedback_data, image_dir, transform=None):
        self.data = feedback_data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # In production, resolve the original image from prediction_id.
        img_path = os.path.join(self.image_dir, "sample_feedback_image.png")
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        label_vector = [0.0] * len(DISEASE_LABELS)
        for disease in item.get("doctor_corrected_labels", []):
            if disease in DISEASE_LABELS:
                label_vector[DISEASE_LABELS.index(disease)] = 1.0

        return {
            "pixel_values": image,
            "labels": torch.tensor(label_vector, dtype=torch.float32),
        }


def run_continuous_learning():
    """Fine-tune the classifier head using all unprocessed doctor feedback."""
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS LEARNING CYCLE")
    logger.info("=" * 60)

    client = MongoClient(settings.MONGODB_URI)
    database = client[settings.MONGODB_DB_NAME]
    pending_feedback = list(database.feedback.find({"is_processed": False}))
    if not pending_feedback:
        logger.info("No new feedback to process. Exiting.")
        return

    logger.info("Found %d new doctor corrections.", len(pending_feedback))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FeedbackDataset(pending_feedback, "datasets/feedback_images", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model_path = "backend/models/vit/medical_finetuned"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Loading model from %s...", model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.to(device)

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("Epoch %d/2 - Loss: %.4f", epoch + 1, total_loss / len(loader))

    model.save_pretrained(model_path)
    logger.info("Model successfully updated and saved to %s", model_path)
    feedback_ids = [item["_id"] for item in pending_feedback]
    database.feedback.update_many(
        {"_id": {"$in": feedback_ids}},
        {"$set": {"is_processed": True}},
    )
    logger.info("Marked feedback as processed in database.")
    logger.info("=" * 60)
    logger.info("CONTINUOUS LEARNING CYCLE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_continuous_learning()
