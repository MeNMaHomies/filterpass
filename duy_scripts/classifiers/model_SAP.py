import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, dropout=0.1):
        """
        Self-Attention Pooling using a learnable query (CLS-style token).

        The model learns to attend over the sequence and extract
        a single aggregated representation.
        """
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)

        Returns:
            pooled: (B, D)
        """
        B = x.size(0)

        # Expand learnable query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, D)

        # Query attends to the full sequence
        attn_out, _ = self.attention(query, x, x)  # (B, 1, D)

        pooled = attn_out.squeeze(1)  # (B, D)
        pooled = self.norm(pooled)

        return pooled


class SAPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        super().__init__()

        print("Initialising Self-Attention Pooling Classifier")

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False

        self.pooling = SelfAttentionPooling(
            input_dim=768,
            num_heads=8,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, 768)

        pooled = self.pooling(hidden)       # (B, 768)
        logits = self.classifier(pooled)    # (B, 2)

        return logits

# class MSAP(nn.Module):
#     def __init__(self, input_dim=768, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.input_dim = input_dim

#         self.linear = nn.Linear(input_dim, input_dim)
#         self.attention = nn.Linear(input_dim, num_heads)

#     def forward(self, x):
#         # x: (B, T, D)
#         B = x.size(0)

#         h = torch.tanh(self.linear(x))           # (B, T, D)
#         scores = self.attention(h)               # (B, T, H)
#         weights = torch.softmax(scores, dim=1)   # (B, T, H)

#         x_exp = x.unsqueeze(2)                   # (B, T, 1, D)
#         weights_exp = weights.unsqueeze(-1)      # (B, T, H, 1)

#         weighted = weights_exp * x_exp           # (B, T, H, D)
#         pooled = weighted.sum(dim=1)             # (B, H, D)

#         pooled = pooled.reshape(B, -1)           # (B, H * D)
#         return pooled


# class SAPClassifier(nn.Module):
#     def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True, num_heads=8):
#         super().__init__()

#         print("Initialising SAP Classifier (Multi-Head Self-Attentive Pooling)")
#         self.encoder = Wav2Vec2Model.from_pretrained(model_name)

#         if freeze_extractor:
#             for param in self.encoder.feature_extractor.parameters():
#                 param.requires_grad = False

#         self.attention_pooling = MSAP(
#             input_dim=768,
#             num_heads=num_heads,
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(768 * num_heads, 256),
#             nn.GELU(),
#             nn.Dropout(0.4),
#             nn.Linear(256, 2),
#         )

#     def forward(self, input_values, attention_mask=None):
#         outputs = self.encoder(input_values, attention_mask=attention_mask)
#         hidden = outputs.last_hidden_state           # (B, T, 768)
#         pooled = self.attention_pooling(hidden)      # (B, 768 * num_heads)
#         logits = self.classifier(pooled)             # (B, 2)
#         return logits
