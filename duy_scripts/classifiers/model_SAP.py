import torch
import torch.nn as nn
from transformers import AutoModel


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

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, T, D)
            attention_mask: (B, T) — 1=valid frame, 0=padding. Optional.

        Returns:
            pooled: (B, D)
        """
        B = x.size(0)

        # Expand learnable query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, D)

        # key_padding_mask: True = positions to IGNORE
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_out, _ = self.attention(
            query, x, x, key_padding_mask=key_padding_mask
        )  # (B, 1, D)

        pooled = attn_out.squeeze(1)  # (B, D)
        pooled = self.norm(pooled)

        return pooled


class SAPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        super().__init__()

        print(f"Initialising Self-Attention Pooling Classifier (backbone={model_name})")

        # AutoModel handles wav2vec2, WavLM, HuBERT etc. without code changes;
        # all expose `feature_extractor`, `_get_feature_vector_attention_mask`,
        # and `last_hidden_state` so the rest of this class is backbone-agnostic.
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False

        hidden_dim = self.encoder.config.hidden_size

        self.pooling = SelfAttentionPooling(
            input_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T_frames, 768)

        # attention_mask is at sample resolution; downsample to frame resolution
        # to match `hidden`. wav2vec2 stride collapses ~320 samples per frame.
        frame_mask = None
        if attention_mask is not None:
            frame_mask = self.encoder._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )

        pooled = self.pooling(hidden, frame_mask)  # (B, 768)
        logits = self.classifier(pooled)           # (B, 2)

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
