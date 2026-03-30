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

        self.embed = nn.Linear(1024, emb_dim)

    def forward(self, x):
        f = self.backbone(x)
        z = self.embed(f)
        z = F.normalize(z, p=2, dim=1)
        return z


# ==========================
# Feature Extraction
# ==========================

@torch.no_grad()
def get_spatial_vectors(spatial_np, model, device):

    dataset = SpatialDataset(spatial_np)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model.eval()
    vecs = []

    for x in loader:
        x = x.to(device)
        z = model(x)
        vecs.append(z.cpu())

    return torch.cat(vecs)

