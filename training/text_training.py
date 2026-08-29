# training/text_training.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging

# Ensure backend is in sys.path to import the model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from backend.models.clinical_model import ClinicalTransformerModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ClinicalTextDataset(Dataset):
    """
    Swappable Dataset for Clinical Text (Symptoms).
    
    TODO: Plug in your real dataset here (e.g., MIMIC-III, custom symptom CSV).
    Expected directory structure:
        datasets/
            symptoms.csv  # Columns: 'text' (symptom description), 'label' (integer class)
    """
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(
    model: ClinicalTransformerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 2e-5,
    device: str = "cuda"
):
    """
    Training loop for the Clinical Transformer model.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
                
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")
        
        # TODO: Save checkpoint
        # os.makedirs("../models/clinical", exist_ok=True)
        # torch.save(model.state_dict(), f"../models/clinical/checkpoint_epoch_{epoch}.pth")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Training script ready. Uncomment dataset loading and train_model() to begin training.")
    
    # TODO: Replace with actual data loading logic
    # import pandas as pd
    # df = pd.read_csv("../datasets/symptoms.csv")
    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # dataset = ClinicalTextDataset(df['text'].tolist(), df['label'].tolist(), tokenizer)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    # model = ClinicalTransformerModel(num_labels=14)
    # train_model(model, train_loader, val_loader, epochs=5)
