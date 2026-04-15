import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class MSAP(nn.Module):
    def __init__(self, input_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        self.linear = nn.Linear(input_dim, input_dim)
        self.attention = nn.Linear(input_dim, num_heads)

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)

        h = torch.tanh(self.linear(x))  # (B, T, D)
        scores = self.attention(h)  # (B, T, H)
        weights = torch.softmax(scores, dim=1)  # (B, T, H)

        x_exp = x.unsqueeze(2)  # (B, T, 1, D)
        weights_exp = weights.unsqueeze(-1)  # (B, T, H, 1)

        weighted = weights_exp * x_exp  # (B, T, H, D)
        pooled = weighted.sum(dim=1)  # (B, H, D)

        pooled = pooled.reshape(B, -1)  # (B, H * D)
        return pooled


class SAPClassifier(nn.Module):
    def __init__(
        self, model_name="facebook/wav2vec2-base", freeze_extractor=True, num_heads=8
    ):
        super().__init__()

        print("Initialising SAP Classifier (Multi-Head Self-Attentive Pooling)")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False

        self.attention_pooling = MSAP(
            input_dim=768,
            num_heads=num_heads,
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 * num_heads, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, 768)
        pooled = self.attention_pooling(hidden)  # (B, 768 * num_heads)
        logits = self.classifier(pooled)  # (B, 2)
        return logits
