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

        self.backbone = densenet121(weights=None)
        
        import os
        import re
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'densenet121.pth')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Fix keys due to PyTorch versioning differences
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
            )
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            
            self.backbone.load_state_dict(state_dict)
        else:
            print(f"Warning: {weights_path} not found! Spatial features will be random.")

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