import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================
# Dataset
# ==========================

class FrequencyDataset(Dataset):
    def __init__(self, freq_np, labels_np):
        # (N, H, W, 1) → (N, 1, H, W)
        self.x = torch.tensor(freq_np, dtype=torch.float32).permute(0,3,1,2)
        self.y = torch.tensor(labels_np, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==========================
# Improved CNN (Frequency-Aware)
# ==========================

class FrequencyCNN(nn.Module):
    def __init__(self, num_classes=2, emb_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4 (important for deeper features)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.embed = nn.Linear(256, emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        z = self.embed(f)

        logits = self.cls(z)
        return logits, F.normalize(z, p=2, dim=1)


# ==========================
# Training
# ==========================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ==========================
# Feature Extraction
# ==========================

def get_frequency_vectors(freq_np, labels_np, device, epochs=5):

    dataset = FrequencyDataset(freq_np, labels_np)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = FrequencyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for e in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"[Freq] Epoch {e+1}: {loss:.4f}")

    # Extract embeddings (no grad needed here)
    model.eval()
    vecs = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, z = model(x)
            vecs.append(z.cpu())

    return torch.cat(vecs), model


def extract_frequency_features(freq_np, model, device):
    dummy_labels = np.zeros(len(freq_np))
    dataset = FrequencyDataset(freq_np, dummy_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model.eval()
    vecs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, z = model(x)
            vecs.append(z.cpu())
            
    return torch.cat(vecs)