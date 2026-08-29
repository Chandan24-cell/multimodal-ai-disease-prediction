# training/image_training.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging

# Ensure backend is in sys.path to import the model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from models.vit_model import MedicalViTModel

logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """
    Swappable Dataset for Medical Images.
    
    TODO: Plug in your real dataset here (e.g., NIH ChestX-ray14, CheXpert, or Kaggle MRI).
    Expected directory structure:
        datasets/
            images/
                img1.png
                img2.png
            labels.csv  # Columns: 'filename', 'label_0', 'label_1', ...
    """
    def __init__(self, image_dir: str, labels_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # TODO: Implement pandas read_csv here to load labels
        # import pandas as pd
        # self.data = pd.read_csv(labels_file)
        self.data = [] # Placeholder
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: Replace with actual data loading logic
        # img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        # image = Image.open(img_name).convert('RGB')
        # if self.transform: image = self.transform(image)
        # labels = self.data.iloc[idx, 1:].values.astype(float)
        # return image, torch.tensor(labels, dtype=torch.float32)
        
        # Placeholder return for syntax validation
        dummy_image = torch.randn(3, 224, 224)
        dummy_label = torch.randint(0, 2, (14,)).float() # Multi-label example
        return dummy_image, dummy_label

def train_model(
    model: MedicalViTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 2e-5,
    device: str = "cuda"
):
    """
    Training loop for the ViT model.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # For multi-label classification, use BCEWithLogitsLoss
    # For multi-class (single label), use CrossEntropyLoss
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
                
        logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {running_loss/len(train_loader):.4f}")
        
        # TODO: Save checkpoint
        # os.makedirs("../models/vit", exist_ok=True)
        # torch.save(model.state_dict(), f"../models/vit/checkpoint_epoch_{epoch}.pth")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # TODO: Update these paths to your actual dataset
    IMAGE_DIR = "../datasets/images"
    LABELS_FILE = "../datasets/labels.csv"
    
    logger.info("Training script ready. Uncomment dataset loading and train_model() to begin training.")
    # dataset = MedicalImageDataset(IMAGE_DIR, LABELS_FILE)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    # model = MedicalViTModel(num_labels=14)
    # train_model(model, train_loader, val_loader, epochs=5)