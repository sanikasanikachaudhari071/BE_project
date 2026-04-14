import os
import glob
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Transformer.transfromermodel import FusionTransformer

# ==========================
# REPRODUCIBILITY
# ==========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DATA LOADING (PRE-COMPUTED EMBEDDINGS)
# ==========================
BASE_DIR = "/content/embeddings"
FAKE_DIR = os.path.join(BASE_DIR, "fake")
REAL_DIR = os.path.join(BASE_DIR, "real")

# Get all cached embedding files
fake_files = glob.glob(os.path.join(FAKE_DIR, "*.pt"))
real_files = glob.glob(os.path.join(REAL_DIR, "*.pt"))

print(f"Total FAKE sequences: {len(fake_files)}")
print(f"Total REAL sequences: {len(real_files)}")

# Balance the dataset (under-sample FAKE)
min_samples = min(len(fake_files), len(real_files))

if min_samples == 0:
    print("No localized embeddings found. Please run scripts `data_pipeline/1_extract_faces.py` and `2_cache_embeddings.py` first.")
    exit(1)

fake_files = random.sample(fake_files, min_samples)
real_files = random.sample(real_files, min_samples)

all_files = fake_files + real_files
labels = [1]*len(fake_files) + [0]*len(real_files)

# Video-level splitting (no data leakage)
X_train_val_files, X_test_files, y_train_val, y_test = train_test_split(
    all_files, labels, test_size=0.2, random_state=SEED, stratify=labels
)
X_train_files, X_val_files, y_train, y_val = train_test_split(
    X_train_val_files, y_train_val, test_size=0.1, random_state=SEED, stratify=y_train_val
)

print(f"Train/Val/Test Split: {len(X_train_files)} / {len(X_val_files)} / {len(X_test_files)} Videos")


# ==========================
# DATASET
# ==========================
class EmbeddingDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location=device)
        return data["spatial"], data["freq"], data["label"]

# Custom collate_fn since sequences can have different lengths
# i.e., different numbers of frames extracted
def pad_collate(batch):
    sp_list, fr_list, labels_list = [], [], []
    for sp, fr, lbl in batch:
        sp_list.append(sp)
        fr_list.append(fr)
        labels_list.append(lbl)

    # Pad sequences to match the max length in batch
    sp_padded = torch.nn.utils.rnn.pad_sequence(sp_list, batch_first=True)
    fr_padded = torch.nn.utils.rnn.pad_sequence(fr_list, batch_first=True)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return sp_padded, fr_padded, labels

train_loader = DataLoader(EmbeddingDataset(X_train_files), batch_size=16, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(EmbeddingDataset(X_val_files), batch_size=16, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(EmbeddingDataset(X_test_files), batch_size=16, shuffle=False, collate_fn=pad_collate)


# ==========================
# TRANSFORMER TRAINING
# ==========================
model = FusionTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 40
PATIENCE = 5
best_val_loss = float('inf')
early_stop_counter = 0

print("\n--- TRAINING FUSION MODEL ---")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for sp, fr, lbl in train_loader:
        sp, fr, lbl = sp.to(device), fr.to(device), lbl.to(device)
        optimizer.zero_grad()

        outputs = model(sp, fr).squeeze()
        if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            
        loss = criterion(outputs, lbl)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sp, fr, lbl in val_loader:
            sp, fr, lbl = sp.to(device), fr.to(device), lbl.to(device)
            outputs = model(sp, fr).squeeze()
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, lbl)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f}  |  Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "deepfake_model.pth")
    else:
        early_stop_counter += 1
        
    if early_stop_counter >= PATIENCE:
        print(f"\n[!] Early Stopping Triggered! Validation loss failed to improve for {PATIENCE} epochs.")
        break

# ==========================
# EVALUATION (TEST SET)
# ==========================
if os.path.exists("deepfake_model.pth"):
    model.load_state_dict(torch.load("deepfake_model.pth"))
    
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for sp, fr, lbl in test_loader:
        sp, fr = sp.to(device), fr.to(device)
        outputs = model(sp, fr).squeeze()
        if outputs.dim() == 0: outputs = outputs.unsqueeze(0)

        preds = (torch.sigmoid(outputs) > 0.6).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(lbl.numpy())

print("\n===== Evaluationing Testing =====")
print("Accuracy :", f"{accuracy_score(all_true, all_preds) * 100:.2f}%")
print("Precision:", f"{precision_score(all_true, all_preds, zero_division=0) * 100:.2f}%")
print("Recall   :", f"{recall_score(all_true, all_preds, zero_division=0) * 100:.2f}%")
print("F1 Score :", f"{f1_score(all_true, all_preds, zero_division=0) * 100:.2f}%")
print("\nModel trained using cached embeddings and evaluated successfully!")