import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    def __init__(self, spatial_dim=1024, freq_dim=128, emb_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        self.emb_dim = emb_dim
        
        # Projection layers to map inputs to common embedding dimension
        self.spatial_proj = nn.Linear(spatial_dim, emb_dim)
        self.freq_proj = nn.Linear(freq_dim, emb_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(2, emb_dim))  
        # 2 tokens → spatial + frequency

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, spatial_vec, freq_vec):
        """
        spatial_vec: (B, T, spatial_dim) or (B, spatial_dim)
        freq_vec:    (B, T, freq_dim) or (B, freq_dim)
        """
        # Average across the frames/sequence dimension (T) so we are left with a single representation per video
        if spatial_vec.dim() == 3:
            spatial_vec = spatial_vec.mean(dim=1)
        if freq_vec.dim() == 3:
            freq_vec = freq_vec.mean(dim=1)
            
        # Project inputs to common embedding dimension
        sp_emb = self.spatial_proj(spatial_vec)
        fr_emb = self.freq_proj(freq_vec)

        # Stack as sequence (B, 2, emb_dim)
        x = torch.stack([sp_emb, fr_emb], dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Pool (take mean of tokens)
        x = x.mean(dim=1)

        # Classification
        out = self.classifier(x)

        return out