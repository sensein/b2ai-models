from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch

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

        # decide whether to chunk
        if waveform.shape[0] > self.max_seconds * self.sr:
            waveforms = self._chunk_audio(waveform)
        else:
            waveforms = [waveform]

        # feature extraction per chunk
        inputs = []
        for w in waveforms:
            feats = self.feature_extractor(
                w.numpy(),
                sampling_rate=self.sr,
                return_tensors="pt"
            ).input_values.squeeze(0)
            inputs.append(feats)

        # stack chunks: (num_chunks, T)
        inputs = torch.stack(inputs)

        return inputs, label