import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SPClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_extractor=True):
        """
        Wav2Vec2 model with a binary classification head for synthetic audio detection.
        
        Args:
            model_name (str): The Hugging Face model repository to load.
            freeze_extractor (bool): If True, freezes the CNN feature extractor layers.
        """
        super().__init__()
        
        print("Initialising Stats Pooling Classfier")
        # Load the base Wav2Vec2 model (without the language modeling head)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze the CNN feature extractor (bottom 7 convolutional layers)
        if freeze_extractor:
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
                
        # Define the classification head
        self.classifier = nn.Sequential(
            nn.Linear(1536, 256),
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
        
        hidden = outputs.last_hidden_state
        

        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1)

        pooled = torch.cat([mean, std], dim=1)
        
        # Pass through the linear classification head
        logits = self.classifier(pooled)
        
        return logits
    
