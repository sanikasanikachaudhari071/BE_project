
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

from preprocessing.preprocess import run_preprocessing_media
from densenet.densenet import SpatialDenseNet, get_spatial_vectors
from frequencycnn.frequency import get_frequency_vectors
from Transformer.transfromermodel import FusionTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD CSV
# ==========================
if os.path.exists("/content/drive"):
    csv_path = "/content/drive/MyDrive/videos.csv"
    base_path = "/content/drive/MyDrive/"
else:
    csv_path = "videos.csv"
    base_path = ""

df = pd.read_csv(csv_path)

# ==========================
# BALANCE DATASET
# ==========================
df_fake = df[df["label"] == 1].sample(100, random_state=42)
df_real = df[df["label"] == 0].sample(100, random_state=42)
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)

# Fix paths
df["video_path"] = df["video_path"].apply(lambda x: base_path + x)

# ==========================
# MODELS
# ==========================
spatial_model = SpatialDenseNet().to(device)

all_spatial = []
all_freq_data = []
all_labels = []

# ==========================
# LOOP OVER DATA
# ==========================
for i, row in df.iterrows():

    print(f"Processing {i+1}/{len(df)}:", row["video_path"])

    path = row["video_path"]
    label = row["label"]

    spatial, freq, _ = run_preprocessing_media(path)

    if spatial is None or freq is None or len(spatial) == 0:
        print("Skipped (no faces)")
        continue

    # -------- SPATIAL --------
    spatial_vec = get_spatial_vectors(spatial, spatial_model, device)

    # -------- STORE --------
    all_spatial.append(spatial_vec)
    all_freq_data.append(freq)
    all_labels.extend([label] * len(spatial_vec))


# ==========================
# CHECK DATA
# ==========================
if len(all_spatial) == 0:
    raise ValueError("No valid data processed.")

# ==========================
# STACK SPATIAL
# ==========================
final_spatial = torch.cat(all_spatial)
print("Spatial:", final_spatial.shape)

# ==========================
# TRAIN FREQUENCY MODEL (ONCE)
# ==========================
freq_np = np.concatenate(all_freq_data)
labels_np = np.array(all_labels)

print("Training Frequency Model...")

final_freq, _ = get_frequency_vectors(
    freq_np,
    labels_np,
    device,
    epochs=15
)

print("Frequency:", final_freq.shape)

# ==========================
# NORMALIZE FEATURES
# ==========================
# final_spatial = F.normalize(final_spatial, dim=1)
# final_freq = F.normalize(final_freq, dim=1)

# ==========================
# BALANCE FEATURES
# ==========================
labels = torch.tensor(all_labels)

fake_idx = (labels == 1).nonzero(as_tuple=True)[0]
real_idx = (labels == 0).nonzero(as_tuple=True)[0]

min_samples = min(len(fake_idx), len(real_idx))

fake_idx = fake_idx[:min_samples]
real_idx = real_idx[:min_samples]

balanced_idx = torch.cat([fake_idx, real_idx])

final_spatial = final_spatial[balanced_idx]
final_freq = final_freq[balanced_idx]
labels = labels[balanced_idx]

print("Balanced samples:", len(labels))

# ==========================
# TRAIN-TEST SPLIT
# ==========================
Xsp_train, Xsp_test, Xf_train, Xf_test, y_train, y_test = train_test_split(
    final_spatial,
    final_freq,
    labels.float(),
    test_size=0.2,
    random_state=42
)

# ==========================
# DATASET
# ==========================
class FusionDataset(Dataset):
    def __init__(self, sp, fr, labels):
        self.sp = sp
        self.fr = fr
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sp[idx], self.fr[idx], self.labels[idx]


train_loader = DataLoader(FusionDataset(Xsp_train, Xf_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(FusionDataset(Xsp_test, Xf_test, y_test), batch_size=32)

# ==========================
# FUSION MODEL (TRANSFORMER)
# ==========================
model = FusionTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 10

# ==========================
# TRAINING
# ==========================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for sp, fr, lbl in train_loader:
        sp, fr, lbl = sp.to(device), fr.to(device), lbl.to(device)

        optimizer.zero_grad()

        outputs = model(sp, fr).squeeze()
        loss = criterion(outputs, lbl)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# ==========================
# EVALUATION
# ==========================
model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for sp, fr, lbl in test_loader:

        sp, fr = sp.to(device), fr.to(device)

        outputs = model(sp, fr).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(lbl.numpy())

print("\n===== Evaluation =====")
print("Accuracy :", accuracy_score(all_true, all_preds))
print("Precision:", precision_score(all_true, all_preds, zero_division=0))
print("Recall   :", recall_score(all_true, all_preds, zero_division=0))
print("F1 Score :", f1_score(all_true, all_preds, zero_division=0))

# ==========================
# SAVE MODEL
# ==========================
torch.save(model.state_dict(), "deepfake_model.pth")
print("\nModel saved successfully!")