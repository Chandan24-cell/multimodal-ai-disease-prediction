import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MedicalViTModel(nn.Module):

    def __init__(self, num_labels=14, model_name=None):
        super().__init__()

        self.num_labels = num_labels

        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        self.embedding_dim = 1024

        self.backbone.classifier[3] = nn.Linear(
            self.embedding_dim,
            num_labels
        )

    def forward(self, pixel_values, labels=None):

        features = self.backbone.features(pixel_values)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        logits = self.backbone.classifier(features)

        return type(
            "Output",
            (),
            {"logits": logits}
        )()

    def get_last_hidden_state(self, pixel_values):

        with torch.no_grad():
            features = self.backbone.features(pixel_values)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)

        return features.unsqueeze(1)
