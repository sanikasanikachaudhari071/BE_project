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
    # Running in Colab
    csv_path = "/content/drive/MyDrive/videos.csv"
    base_path = "/content/drive/MyDrive/"
else:
    # Running locally
    csv_path = "videos.csv"
    base_path = ""

df = pd.read_csv(csv_path)

# Fix paths
df["video_path"] = df["video_path"].apply(
    lambda x: base_path + x
)

spatial_model = SpatialDenseNet().to(device)

all_spatial = []
all_freq_data = []
all_labels = []

# ==========================
# LOOP OVER DATA
# ==========================
for _, row in df.iterrows():

    path = row["video_path"]
    label = row["label"]

    spatial, freq, _ = run_preprocessing_media(path)

    if spatial is None or freq is None:
        continue

    # -------- SPATIAL --------
    spatial_vec = get_spatial_vectors(spatial, spatial_model, device)
    all_spatial.append(spatial_vec)

    # -------- FREQUENCY --------
    all_freq_data.append(freq)

    # -------- LABELS --------
    all_labels.extend([label] * len(spatial_vec))


# ==========================
# STACK SPATIAL
# ==========================
final_spatial = torch.cat(all_spatial)

print("Spatial:", final_spatial.shape)


# ==========================
# TRAIN FREQUENCY ONCE
# ==========================
freq_np = np.concatenate(all_freq_data)
labels_np = np.array(all_labels)

final_freq, _ = get_frequency_vectors(
    freq_np,
    labels_np,
    device,
    epochs=5
)

print("Frequency:", final_freq.shape)


# ==========================
# TRAIN-TEST SPLIT
# ==========================
X_spatial = final_spatial
X_freq = final_freq
y = torch.tensor(all_labels, dtype=torch.float32)

Xsp_train, Xsp_test, Xf_train, Xf_test, y_train, y_test = train_test_split(
    X_spatial, X_freq, y, test_size=0.2, random_state=42
)


# ==========================
# DATASET + LOADER
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
# TRAIN TRANSFORMER
# ==========================
model = FusionTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for sp, fr, labels in train_loader:
        sp, fr, labels = sp.to(device), fr.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(sp, fr).squeeze()
        loss = criterion(outputs, labels)

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
    for sp, fr, labels in test_loader:

        sp, fr = sp.to(device), fr.to(device)

        outputs = model(sp, fr).squeeze()
        preds = (outputs > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(labels.numpy())


print("\n===== Evaluation =====")
print("Accuracy :", accuracy_score(all_true, all_preds))
print("Precision:", precision_score(all_true, all_preds))
print("Recall   :", recall_score(all_true, all_preds))
print("F1 Score :", f1_score(all_true, all_preds))


# ==========================
# SAVE MODEL
# ==========================
torch.save(model.state_dict(), "deepfake_model.pth")
print("\nModel saved as deepfake_model.pth")