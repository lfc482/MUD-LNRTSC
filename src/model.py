import copy
import torch
import torch.nn as nn


class CNNLSTMRecon(nn.Module):
    """CNN-LSTM classifier with a lightweight auxiliary reconstruction head."""

    def __init__(self, input_dim: int, num_classes: int = 2, conv_channels: int = 64,
                 lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)
        self.recon_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, input_dim),
        )

    def forward(self, x):
        # x: [B, T, F]
        z = x.transpose(1, 2)       # [B, F, T]
        z = self.conv(z)            # [B, C, T]
        z = z.transpose(1, 2)       # [B, T, C]
        out, _ = self.lstm(z)       # [B, T, H]
        h = self.dropout(out[:, -1, :])
        logits = self.classifier(h)
        recon = self.recon_head(out)
        return logits, recon, h


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


def copy_model(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher
