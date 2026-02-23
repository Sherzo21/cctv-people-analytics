import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image


class ResNet50WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2),
        )

    def forward(self, x):
        return self.base(x)


class GenderClassifier:
    """
    Matches your training preprocessing:
      - Resize(224,224)
      - ToTensor()
      - NO normalization
    """

    def __init__(self, weights_path: str, device: str | None = None, label_map=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet50WithDropout().to(self.device)

        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # IMPORTANT: to confirm if 0=male,1=female (or opposite)
        self.label_map = label_map or {0: "male", 1: "female"}

    @torch.no_grad()
    def predict(self, bgr_crop: np.ndarray):
        if bgr_crop is None or bgr_crop.size == 0:
            return "unknown", 0.0

        rgb = bgr_crop[:, :, ::-1]
        pil = Image.fromarray(rgb)

        x = self.tf(pil).unsqueeze(0).to(self.device)  # [1,3,224,224]
        logits = self.model(x)  # [1,2]
        probs = torch.softmax(logits, dim=1).squeeze(0)

        conf, pred = torch.max(probs, dim=0)
        label = self.label_map[int(pred.item())]
        return label, float(conf.item()), probs.cpu().numpy()