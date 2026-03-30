# import pandas as pd
# import torch
# import numpy as np
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from torch.utils.data import Dataset, DataLoader

# from preprocessing.preprocess import run_preprocessing_media
# from densenet.densenet import SpatialDenseNet, get_spatial_vectors
# from frequencycnn.frequency import get_frequency_vectors
# from Transformer.transfromermodel import FusionTransformer


# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ==========================
# # LOAD CSV
# # ==========================
# import os

# if os.path.exists("/content/drive"):
#     # Running in Colab
#     csv_path = "/content/drive/MyDrive/videos.csv"
#     base_path = "/content/drive/MyDrive/"
# else:
#     # Running locally
#     csv_path = "videos.csv"
#     base_path = ""

# df = pd.read_csv(csv_path)

# # Fix paths
# df["video_path"] = df["video_path"].apply(
#     lambda x: base_path + x
# )

# spatial_model = SpatialDenseNet().to(device)

# all_spatial = []
# all_freq_data = []
# all_labels = []

# # ==========================
# # LOOP OVER DATA
# # ==========================
# for _, row in df.iterrows():

#     path = row["video_path"]
#     label = row["label"]

#     spatial, freq, _ = run_preprocessing_media(path)

#     if spatial is None or freq is None:
#         continue

#     # -------- SPATIAL --------
#     spatial_vec = get_spatial_vectors(spatial, spatial_model, device)
#     all_spatial.append(spatial_vec)

#     # -------- FREQUENCY --------
#     all_freq_data.append(freq)

#     # -------- LABELS --------
#     all_labels.extend([label] * len(spatial_vec))


# # ==========================
# # STACK SPATIAL
# # ==========================
# final_spatial = torch.cat(all_spatial)

# print("Spatial:", final_spatial.shape)


# # ==========================
# # TRAIN FREQUENCY ONCE
# # ==========================
# freq_np = np.concatenate(all_freq_data)
# labels_np = np.array(all_labels)

# final_freq, _ = get_frequency_vectors(
#     freq_np,
#     labels_np,
#     device,
#     epochs=5
# )

# print("Frequency:", final_freq.shape)


# # ==========================
# # TRAIN-TEST SPLIT
# # ==========================
# X_spatial = final_spatial
# X_freq = final_freq
# y = torch.tensor(all_labels, dtype=torch.float32)

# Xsp_train, Xsp_test, Xf_train, Xf_test, y_train, y_test = train_test_split(
#     X_spatial, X_freq, y, test_size=0.2, random_state=42
# )


# # ==========================
# # DATASET + LOADER
# # ==========================
# class FusionDataset(Dataset):
#     def __init__(self, sp, fr, labels):
#         self.sp = sp
#         self.fr = fr
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.sp[idx], self.fr[idx], self.labels[idx]


# train_loader = DataLoader(FusionDataset(Xsp_train, Xf_train, y_train), batch_size=32, shuffle=True)
# test_loader = DataLoader(FusionDataset(Xsp_test, Xf_test, y_test), batch_size=32)


# # ==========================
# # TRAIN TRANSFORMER
# # ==========================
# model = FusionTransformer().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCELoss()

# EPOCHS = 10

# for epoch in range(EPOCHS):

#     model.train()
#     total_loss = 0

#     for sp, fr, labels in train_loader:
#         sp, fr, labels = sp.to(device), fr.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(sp, fr).squeeze()
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


# # ==========================
# # EVALUATION
# # ==========================
# model.eval()

# all_preds = []
# all_true = []

# with torch.no_grad():
#     for sp, fr, labels in test_loader:

#         sp, fr = sp.to(device), fr.to(device)

#         outputs = model(sp, fr).squeeze()
#         preds = (outputs > 0.5).cpu().numpy()

#         all_preds.extend(preds)
#         all_true.extend(labels.numpy())


# print("\n===== Evaluation =====")
# print("Accuracy :", accuracy_score(all_true, all_preds))
# print("Precision:", precision_score(all_true, all_preds))
# print("Recall   :", recall_score(all_true, all_preds))
# print("F1 Score :", f1_score(all_true, all_preds))


# # ==========================
# # SAVE MODEL
# # ==========================
# torch.save(model.state_dict(), "deepfake_model.pth")
# print("\nModel saved as deepfake_model.pth")

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

from preprocessing.preprocess import run_preprocessing_media
from densenet.densenet import SpatialDenseNet, get_spatial_vectors
from frequencycnn.frequency import get_frequency_vectors
from Transformer.transfromermodel import FusionTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD CSV
# ==========================
import os

if os.path.exists("/content/drive"):
    csv_path = "/content/drive/MyDrive/videos.csv"
    base_path = "/content/drive/MyDrive/"
else:
    csv_path = "videos.csv"
    base_path = ""

df = pd.read_csv(csv_path)

# ==========================
# 🔥 BALANCE VIDEO DATASET
# ==========================
df_fake = df[df["label"] == 1].sample(75, random_state=42)
df_real = df[df["label"] == 0].sample(75, random_state=42)
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)

# Fix paths
df["video_path"] = df["video_path"].apply(
    lambda x: base_path + x
)

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

# 🔥 FILTER BAD DATA
    if spatial is None or freq is None or len(spatial) == 0:
        print("Skipped (too few faces)")
        continue

    # -------- SPATIAL --------
    spatial_vec = get_spatial_vectors(spatial, spatial_model, device)
    all_spatial.append(spatial_vec)

    # -------- FREQUENCY --------
    all_freq_data.append(freq)

    # -------- LABELS --------
    all_labels.extend([label] * len(spatial_vec))


# ==========================
# CHECK DATA
# ==========================
if len(all_spatial) == 0:
    raise ValueError("No valid data processed.")

# ==========================
# STACK FEATURES
# ==========================
final_spatial = torch.cat(all_spatial)
print("Spatial:", final_spatial.shape)

freq_np = np.concatenate(all_freq_data)
labels_np = np.array(all_labels)

# ==========================
# TRAIN FREQUENCY MODEL
# ==========================
final_freq, _ = get_frequency_vectors(
    freq_np,
    labels_np,
    device,
    epochs=10
)

print("Frequency:", final_freq.shape)
# 🔥 NORMALIZE FEATURES
import torch.nn.functional as F

final_spatial = F.normalize(final_spatial, dim=1)
final_freq = F.normalize(final_freq, dim=1)

# ==========================
# 🔥 FEATURE BALANCING (VERY IMPORTANT)
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
# MODEL
# ==========================
class SimpleFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, sp, fr):
        x = torch.cat([sp, fr], dim=1)
        return self.classifier(x)
model = SimpleFusion().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)   # 🔥 lower LR
pos_weight = torch.tensor([1.5]).to(device)   # 🔥 adjust if needed
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

EPOCHS = 10   # 🔥 more training

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

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

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
# torch.save(model.state_dict(), "deepfake_model.pth")
# print("\nModel saved successfully!")