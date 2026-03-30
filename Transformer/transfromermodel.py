import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        self.emb_dim = emb_dim

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
            nn.Linear(64, 1)
        )

    def forward(self, spatial_vec, freq_vec):
        """
        spatial_vec: (B, 128)
        freq_vec:    (B, 128)
        """

        # Stack as sequence (B, 2, 128)
        x = torch.stack([spatial_vec, freq_vec], dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Pool (take mean of tokens)
        x = x.mean(dim=1)

        # Classification
        out = self.classifier(x)

        return out