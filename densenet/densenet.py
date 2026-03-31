import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================
# Dataset
# ==========================

class SpatialDataset(Dataset):
    def __init__(self, spatial_np):
        x = torch.tensor(spatial_np, dtype=torch.float32).permute(0,3,1,2)

        # Normalize (important)
        self.x = (x - 0.5) / 0.5

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# ==========================
# DenseNet Model
# ==========================

from torchvision.models import densenet121, DenseNet121_Weights

class SpatialDenseNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()

        # 🔥 FREEZE BACKBONE
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)
        return f
# ==========================
# Feature Extraction
# ==========================
def get_spatial_vectors(spatial_np, model, device):

    dataset = SpatialDataset(spatial_np)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model.eval()
    vecs = []

    for x in loader:
        x = x.to(device)
        z = model(x)
        vecs.append(z.detach().cpu())   # 🔥 detach instead

    return torch.cat(vecs)