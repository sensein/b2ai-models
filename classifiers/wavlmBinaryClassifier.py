
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel

class WavLMBinaryClassifier(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling over time
        logits = self.classifier(pooled)
        return logits.squeeze(-1)  # (B,)

