import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        """
        Learns to assign a weight to each frame based on its content.
        """
        super().__init__()
        # MLP to calculate attention scores
        self.W = nn.Linear(input_dim, 256)
        self.tanh = nn.Tanh()
        self.V = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Output from Wav2Vec2, shape (Batch, Time, Dim)
        Returns:
            torch.Tensor: Attention-weighted pooled representation, shape (Batch, Dim)
        """
        # Calculate raw attention scores
        # Shape: (Batch, Time, 1)
        attn_scores = self.V(self.tanh(self.W(x))) 
        
        # Normalize scores to probabilities across the time dimension
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Multiply each frame by its attention weight and sum across time
        # x * attn_weights relies on broadcasting: (B, T, D) * (B, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1) 
        
        return pooled

class SAPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        """
        Wav2Vec2 model with an attention-based classification head for synthetic audio detection.
        
        Args:
            model_name (str): The Hugging Face model repository to load.
            freeze_extractor (bool): If True, freezes the CNN feature extractor layers.
        """
        super().__init__()
        
        print("Initialising Self Attention Pooling Classfier")
        # Load the base Wav2Vec2 model
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze the CNN feature extractor
        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
                
        # Replace mean pooling with our new attention mechanism
        # Wav2Vec2-base has a hidden dimension of 768
        self.attention_pooling = SelfAttentionPooling(input_dim=768)
                
        # Define the classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values (torch.Tensor): Raw audio waveform, shape (Batch, SequenceLength)
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding.
            
        Returns:
            torch.Tensor: Logits of shape (Batch, 2)
        """
        # Pass through Wav2Vec2
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        
        # Extract the last hidden state: shape (Batch, Time, 768)
        hidden = outputs.last_hidden_state
        
        # Apply Self-Attention Pooling instead of mean pooling
        # Shape collapses to (Batch, 768)
        pooled = self.attention_pooling(hidden)
        
        # Pass through the linear classification head
        logits = self.classifier(pooled)
        
        return logits