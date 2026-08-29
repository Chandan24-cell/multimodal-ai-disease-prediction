# training/fusion_training.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from backend.models.fusion_model import MultimodalFusionModel
from models.classifier import DiseaseClassifier

logger = logging.getLogger(__name__)

class MockFusionDataset(Dataset):
    """
    TODO: Replace with actual pre-extracted embeddings from your ViT, ClinicalBERT, and History models.
    In practice, you run your trained unimodal models on the training set, save the 768-dim embeddings,
    and load them here to train the fusion layer efficiently without loading the massive base models.
    """
    def __init__(self, num_samples=1000, modality_dim=768, num_labels=14):
        self.num_samples = num_samples
        self.modality_dim = modality_dim
        self.num_labels = num_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Mock random embeddings
        img_emb = torch.randn(self.modality_dim)
        txt_emb = torch.randn(self.modality_dim)
        hist_emb = torch.randn(self.modality_dim)
        
        # Mock multi-label targets
        labels = torch.randint(0, 2, (self.num_labels,)).float()
        
        return img_emb, txt_emb, hist_emb, labels

def train_fusion_model(epochs=10, lr=1e-4, device="cuda"):
    logger.info("Starting Fusion Training...")
    
    fusion_model = MultimodalFusionModel().to(device)
    classifier = DiseaseClassifier().to(device)
    
    # Combine parameters
    params = list(fusion_model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    dataset = MockFusionDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        fusion_model.train()
        classifier.train()
        running_loss = 0.0
        
        for batch_idx, (img, txt, hist, labels) in enumerate(loader):
            img, txt, hist, labels = img.to(device), txt.to(device), hist.to(device), labels.to(device)
            
            optimizer.zero_grad()
            fused = fusion_model(img, txt, hist)
            outputs = classifier(fused)
            loss = criterion(outputs["logits"], labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
        
    # TODO: Save weights
    # torch.save(fusion_model.state_dict(), "../models/fusion/fusion_weights.pth")
    # torch.save(classifier.state_dict(), "../models/fusion/classifier_weights.pth")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_fusion_model()
