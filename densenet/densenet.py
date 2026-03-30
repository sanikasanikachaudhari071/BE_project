import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from torchvision.models import densenet121, DenseNet121_Weights

# Import your preprocessing
from preprocessing.preprocess import run_preprocessing_media


# ==========================
# Dataset
# ==========================

class SpatialDataset(Dataset):
    def __init__(self, spatial_np):
        # (N, 224, 224, 3) → (N, 3, 224, 224)
        x = torch.tensor(spatial_np, dtype=torch.float32).permute(0, 3, 1, 2)

        # ImageNet normalization (IMPORTANT)
        self.x = (x - 0.5) / 0.5

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# ==========================
# DenseNet Model
# ==========================

class SpatialDenseNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        # Load pretrained DenseNet121
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Remove classifier
        self.backbone.classifier = nn.Identity()

        # Embedding layer
        self.embed = nn.Linear(1024, emb_dim)

    def forward(self, x):
        f = self.backbone(x)              # (B, 1024)
        z = self.embed(f)                 # (B, emb_dim)
        z = F.normalize(z, p=2, dim=1)    # normalize

        return z


# ==========================
# Feature Extraction
# ==========================

@torch.no_grad()
def extract_spatial_vectors(model, loader, device):
    model.eval()
    vecs = []

    for x in loader:
        x = x.to(device)
        z = model(x)
        vecs.append(z.cpu())

    return torch.cat(vecs)


# ==========================
# MAIN FUNCTION (LIKE frequency.py)
# ==========================

def extract_spatial_features(csv_path):

    df = pd.read_csv(csv_path)

    spatial_all = []

    for _, row in df.iterrows():

        path = row["video_path"]

        spatial, freq, _ = run_preprocessing_media(path)

        if spatial is not None:
            spatial_all.append(spatial)

    # Combine all samples
    spatial_np = np.concatenate(spatial_all, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset + Loader
    ds = SpatialDataset(spatial_np)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    # Model
    model = SpatialDenseNet().to(device)

    # Extract vectors
    vectors = extract_spatial_vectors(model, loader, device)

    return vectors, model


# ==========================
# RUN
# ==========================

if __name__ == "__main__":

    vectors, _ = extract_spatial_features("videos.csv")

    print("Spatial vectors shape:", vectors.shape)