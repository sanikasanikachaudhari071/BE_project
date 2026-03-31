import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import os

from preprocessing.preprocess import run_preprocessing_media
from densenet.densenet import SpatialDenseNet, get_spatial_vectors
from frequencycnn.frequency import get_frequency_vectors, extract_frequency_features
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
# BALANCE VIDEO DATASET
# ==========================
df_fake = df[df["label"] == 1].sample(100, random_state=42)
df_real = df[df["label"] == 0].sample(100, random_state=42)
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)

df["video_path"] = df["video_path"].apply(lambda x: base_path + x)


# ==========================
# SPLIT AT VIDEO LEVEL (LEAKAGE FIX)
# ==========================
# This prevents frames from the same video ending up in both Train and Test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

spatial_model = SpatialDenseNet().to(device)

# all_spatial = []
# all_freq_data = []
# all_labels = []

# ==========================
# PROCESSING UTILS
# ==========================
def process_dataframe(df_subset, sp_model, dev):
    all_spatial = []
    all_freq_data = []
    all_labels = []

    for i, row in df_subset.iterrows():
        print(f"Processing:", row["video_path"])
        path = row["video_path"]
        label = row["label"]

        spatial, freq, _ = run_preprocessing_media(path)

        if spatial is None or freq is None or len(spatial) == 0:
            print("Skipped (no faces)")
            continue

        spatial_vec = get_spatial_vectors(spatial, sp_model, dev)

        all_spatial.append(spatial_vec)
        all_freq_data.append(freq)
        all_labels.extend([label] * len(spatial_vec))

    if len(all_spatial) == 0:
        return None, None, None

    # We must explicitly cast tensors to CPU immediately in loops if we run out of memory, 
    # but here we cast at concatenation
    return torch.cat(all_spatial), np.concatenate(all_freq_data), torch.tensor(all_labels)


def balance_frames(spatial, freq, labels):
    fake_idx = (labels == 1).nonzero(as_tuple=True)[0]
    real_idx = (labels == 0).nonzero(as_tuple=True)[0]

    min_samples = min(len(fake_idx), len(real_idx))

    fake_idx = fake_idx[:min_samples]
    real_idx = real_idx[:min_samples]

    balanced_idx = torch.cat([fake_idx, real_idx])
    
    # Shuffle the balanced tensor
    shuffle_mask = torch.randperm(len(balanced_idx))
    balanced_idx = balanced_idx[shuffle_mask]

    return spatial[balanced_idx], freq[balanced_idx.numpy()], labels[balanced_idx]


# ==========================
# EXTRACT TRAIN & TEST DATA
# ==========================
print("\n--- EXTRACTING TRAIN DATA ---")
Xsp_train_unbal, Xf_train_unbal, y_train_unbal = process_dataframe(df_train, spatial_model, device)

print("\n--- EXTRACTING TEST DATA ---")
Xsp_test_unbal, Xf_test_unbal, y_test_unbal = process_dataframe(df_test, spatial_model, device)

# ==========================
# BALANCE FRAMES
# ==========================
Xsp_train, np_f_train, y_train = balance_frames(Xsp_train_unbal, Xf_train_unbal, y_train_unbal)
Xsp_test, np_f_test, y_test = balance_frames(Xsp_test_unbal, Xf_test_unbal, y_test_unbal)

print(f"\nBalanced Train samples: {len(y_train)}")
print(f"Balanced Test samples: {len(y_test)}")


# ==========================
# TRAIN FREQUENCY MODEL
# ==========================
print("\nTraining Frequency Model on Train Split...")
Xf_train, freq_model = get_frequency_vectors(
    np_f_train,
    y_train.numpy(),
    device,
    epochs=15
)

# Use the explicitly extracted function so we don't accidentally train on the test subset
Xf_test = extract_frequency_features(np_f_test, freq_model, device)

print("Train Spatial:", Xsp_train.shape, "Train Freq:", Xf_train.shape)


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


train_loader = DataLoader(FusionDataset(Xsp_train, Xf_train, y_train.float()), batch_size=32, shuffle=True)
test_loader = DataLoader(FusionDataset(Xsp_test, Xf_test, y_test.float()), batch_size=32)

# ==========================
# FUSION MODEL (TRANSFORMER)
# ==========================
model = FusionTransformer().to(device)

# Provided L2 penalty (weight decay) to heavily combat small-dataset overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
print("Accuracy :", f"{accuracy_score(all_true, all_preds) * 100:.2f}%")
print("Precision:", f"{precision_score(all_true, all_preds, zero_division=0) * 100:.2f}%")
print("Recall   :", f"{recall_score(all_true, all_preds, zero_division=0) * 100:.2f}%")
print("F1 Score :", f"{f1_score(all_true, all_preds, zero_division=0) * 100:.2f}%")

# ==========================
# SAVE MODEL
# ==========================
torch.save(model.state_dict(), "deepfake_model.pth")
print("\nModel saved successfully!")