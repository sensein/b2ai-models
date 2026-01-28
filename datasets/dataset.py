from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch

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
