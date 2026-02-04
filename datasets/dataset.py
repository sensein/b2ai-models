from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, files, labels, sr=16000, model_name="microsoft/wavlm-base", max_seconds=120,  chunk_seconds=20, overlap_seconds=5):
        self.files = files
        self.labels = labels
        self.sr = sr
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.max_seconds = max_seconds
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

    def __len__(self):
        return len(self.files)
        
    def _chunk_audio(self, waveform):
        chunk_len = self.chunk_seconds * self.sr
        hop = (self.chunk_seconds - self.overlap_seconds) * self.sr

        chunks = []
        for start in range(0, waveform.shape[0] - chunk_len + 1, hop):
            chunks.append(waveform[start:start + chunk_len])

        # handle very short or tail-only audio
        if len(chunks) == 0:
            chunks.append(waveform)

        return chunks
    

    def __getitem__(self, idx):
        path = str(self.files[idx]).rstrip()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
    
        waveform, orig_sr = torchaudio.load(f"{path}.wav")
    
        # resample if needed
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)
    
        # mono
        waveform = waveform.mean(dim=0)
    
        arr = waveform.numpy()
        arr[~np.isfinite(arr)] = 0.0
        feats = self.feature_extractor(arr, sampling_rate=self.sr, return_tensors="pt").input_values.squeeze(0)
        feats[torch.isnan(feats)] = 0.0
        feats[torch.isinf(feats)] = 0.0
    
        return feats.unsqueeze(0), label  # keep batch dim as 1 for consistency
