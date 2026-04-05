import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class MPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        """
        Wav2Vec2 model with a binary classification head for synthetic audio detection.
        
        Args:
            model_name (str): The Hugging Face model repository to load.
            freeze_extractor (bool): If True, freezes the CNN feature extractor layers.
        """
        super().__init__()
        
        print("Initialising Mean Pooling Classfier")
        # Load the base Wav2Vec2 model (without the language modeling head)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze the CNN feature extractor (bottom 7 convolutional layers)
        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
                
        # Define the classification head
        # Wav2Vec2-base has a hidden dimension of 768 (mean pooling) or 1536 (stats pooling)

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
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding.
            
        Returns:
            torch.Tensor: Logits of shape (Batch, 2)
        """

        # Pass through Wav2Vec2
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        
        # Extract the last hidden state: shape (Batch, Time, 768)
        hidden = outputs.last_hidden_state
        
        # Mean pooling over the time dimension: shape collapses to (Batch, 768)
        # This creates a single utterance-level embedding
        pooled = hidden.mean(dim=1)
        
        # Pass through the linear classification head
        logits = self.classifier(pooled)
        
        return logits
    
