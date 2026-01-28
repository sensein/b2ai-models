
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor
from datasets.dataset import AudioDataset
from classifiers.wavlmBinaryClassifier import WavLMBinaryClassifier
from train.train import train_model

def collate_fn(batch):
    input_values, labels = zip(*batch)
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.stack(labels)
    return input_values, labels


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

    df = pd.read_csv("peds_annotations.csv")  
    files = df["filename"].tolist()
    labels = df["adult_audio"].tolist()

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