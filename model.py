
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor


class AudioDataset(Dataset):
    def __init__(self, files, labels, sr=16000, model_name="microsoft/wavlm-base"):
        self.files = files
        self.labels = labels
        self.sr = sr
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, orig_sr = torchaudio.load(self.files[idx])

        # resample if needed
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)

        waveform = waveform.mean(dim=0)  # mono

        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sr,
            return_tensors="pt"
        )

        return inputs.input_values.squeeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    input_values, labels = zip(*batch)
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.stack(labels)
    return input_values, labels


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

def train_model(model, dataloader, epochs=3, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    # Save Hugging Face style
    model.save_pretrained("adult-detector-wavlm")
    print("Model saved to adult-detector-wavlm/")


def predict_adult(model, audio_path, sr=16000, device="cuda"):
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")

    waveform, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)

    waveform = waveform.mean(dim=0)  # mono

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=sr,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device))
        prob = torch.sigmoid(logits).item()

    return prob


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("annotations.csv")  
    files = df["filename"].tolist()
    labels = df["label"].tolist()

    dataset = AudioDataset(files, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    model = WavLMBinaryClassifier()


    train_model(model, dataloader, epochs=3, lr=1e-4, device=device)

    # Load model for inference
    model_infer = WavLMBinaryClassifier()
    state_dict = torch.load("adult-detector-wavlm/pytorch_model.bin", map_location=device)
    model_infer.load_state_dict(state_dict)
    model_infer.to(device)
    model_infer.eval()

    # Test on new audio
    test_audio = "new_audio.wav"
    probability = predict_adult(model_infer, test_audio, device=device)
    print(f"Adult probability for {test_audio}: {probability:.2f}")
    print("Adult present?", "YES" if probability > 0.5 else "NO")