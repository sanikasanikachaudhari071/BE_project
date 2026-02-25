import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from preprocessing.preprocess import run_preprocessing_media


# ==========================
# Dataset
# ==========================

class FrequencyDataset(Dataset):
    def __init__(self, freq_np, labels_np):
        self.x = torch.tensor(freq_np, dtype=torch.float32).permute(0,3,1,2)
        self.y = torch.tensor(labels_np, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==========================
# CNN Model
# ==========================

class FrequencyCNN(nn.Module):
    def __init__(self, num_classes, emb_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.embed = nn.Linear(256, emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        z = F.normalize(self.embed(f), p=2, dim=1)
        logits = self.cls(z)
        return logits, z


# ==========================
# Training helpers
# ==========================

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()

        logits,_ = model(x)
        loss = F.cross_entropy(logits,y)
        loss.backward()
        opt.step()

        total += loss.item()*x.size(0)

    return total/len(loader.dataset)


@torch.no_grad()
def extract_vectors(model, loader, device):
    model.eval()
    vecs = []

    for x,_ in loader:
        x = x.to(device)
        _,z = model(x)
        vecs.append(z.cpu())

    return torch.cat(vecs)


# ==========================
# MAIN FUNCTION YOU CALL
# ==========================

def train_frequency_model(data_folder, epochs=10):

    freq_all = []

    for file in os.listdir(data_folder):
        path = os.path.join(data_folder,file)

        spatial,freq,_ = run_preprocessing_media(path)

        if freq is not None:
            freq_all.append(freq)

    freq_np = np.concatenate(freq_all,axis=0)

    labels = np.zeros(len(freq_np),dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FrequencyDataset(freq_np,labels)
    loader = DataLoader(ds,batch_size=16,shuffle=True)

    model = FrequencyCNN(num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)

    for e in range(epochs):
        loss = train_one_epoch(model,loader,opt,device)
        print(f"Epoch {e+1}: {loss:.4f}")

    vectors = extract_vectors(model,loader,device)

    return vectors, model