import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=768):
        """
        Standard Self-Attentive Pooling (SAP).

        Learns a scalar attention score for each time step, then uses
        a weighted sum to produce one utterance-level embedding.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Tensor of shape (B, T, D)
            attention_mask: Optional tensor of shape (B, T)
                            1 = valid, 0 = padding

        Returns:
            Tensor of shape (B, D)
        """
        # Compute attention scores for each frame
        scores = self.attention(x).squeeze(-1)  # (B, T)

        # Mask padded positions if attention_mask is provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=1)  # (B, T)

        # Weighted sum of frame-level features
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)

        return pooled


class SelfAttentivePoolingClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        """
        Wav2Vec2 + Self-Attentive Pooling + classifier head.

        Args:
            model_name (str): Hugging Face model name
            freeze_extractor (bool): If True, freezes only the CNN feature extractor
        """
        super().__init__()

        print("Initialising SAP Classifier")

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False

        self.attention_pooling = SelfAttentionPooling(input_dim=768)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: Raw waveform tensor, shape (B, SeqLen)
            attention_mask: Optional padding mask, shape (B, SeqLen)

        Returns:
            Logits of shape (B, 2)
        """
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, 768)

        # If needed, you may pass a reduced mask aligned with hidden length.
        # For now, SAP can also run without mask.
        pooled = self.attention_pooling(hidden)  # (B, 768)

        logits = self.classifier(pooled)  # (B, 2)
        return logits