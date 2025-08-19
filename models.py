import torch
import torch.nn as nn


class HSMOTE(nn.Module):
    """
    Lightweight dual encoder–decoder:
      N: node encoder   X_n (d_in) -> Z_n (d_embed)
      E: edge encoder   X_e (d_in) -> Z_e (d_embed)
      S: bilinear weight; predict \hat{H} = σ(norm(Z_n) S norm(Z_e)^T)
      D: node decoder   Z_n -> \hat{X}_n
    """
    def __init__(self, d_in=512, d_hid=32, d_embed=32):
        super().__init__()
        self.encoder_node = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_embed)
        )
        self.encoder_edge = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_embed)
        )
        self.S = nn.Parameter(torch.eye(d_embed))
        self.decoder_node = nn.Sequential(
            nn.Linear(d_embed, d_hid), nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_in)
        )

    def predict_H(self, Z_n: torch.Tensor, Z_e: torch.Tensor) -> torch.Tensor:
        """
        Compute \hat{H} = σ(norm(Z_n) S norm(Z_e)^T).
        Z_n: [n_b, d_embed], Z_e: [m_b, d_embed].
        Returns: \hat{H} in (0,1)^{n_b × m_b}.
        """
        Zn = torch.nn.functional.normalize(Z_n, dim=1, eps=1e-8)
        Ze = torch.nn.functional.normalize(Z_e, dim=1, eps=1e-8)
        scores = Zn @ self.S @ Ze.t()
        return torch.sigmoid(scores)


class LinearHead(nn.Module):
    """Linear classifier head: Z_merge -> logits over C classes."""
    def __init__(self, d_embed: int, C: int):
        super().__init__()
        self.fc = nn.Linear(d_embed, C)

    def forward(self, x):
        return self.fc(x)
