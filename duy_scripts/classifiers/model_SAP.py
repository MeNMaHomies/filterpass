import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, dropout=0.1):
        """
        Multi-head attention pooling to capture diverse temporal patterns.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Use mean of sequence as query to attend over all frames
        query = x.mean(dim=1, keepdim=True)         # (B, 1, 768)
        attn_out, _ = self.attention(query, x, x)   # (B, 1, 768)
        attn_out = self.norm(attn_out.squeeze(1))    # (B, 768)
        return attn_out


class SAPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        """
        Wav2Vec2 with multi-head attention pooling and a deeper classifier head.

        Args:
            model_name (str):        HuggingFace model repo to load.
            freeze_extractor (bool): If True, freezes the entire encoder initially.
        """
        super().__init__()

        print("Initialising SAP Classifier (Multi-Head Attention Pooling)")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_extractor:
            # Freeze only the CNN feature extractor, NOT the transformer layers
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False

        # Multi-head attention pooling — captures more diverse temporal patterns
        # than single-head self-attention, better suited for distorted/augmented audio
        self.attention_pooling = MultiHeadAttentionPooling(
            input_dim=768,
            num_heads=8,
            dropout=0.1,
        )

        # Deeper classifier head with residual-style skip connection
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values (torch.Tensor): Raw waveform, shape (Batch, SeqLen)
            attention_mask (torch.Tensor, optional): Padding mask.

        Returns:
            torch.Tensor: Logits of shape (Batch, 2)
        """
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state          # (B, T, 768)
        pooled = self.attention_pooling(hidden)     # (B, 768)
        logits = self.classifier(pooled)            # (B, 2)
        return logits