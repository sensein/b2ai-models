import pandas as pd
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)
import matplotlib.pyplot as plt


model_path = "./peds_voice_model"
model = AutoModelForAudioClassification.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


#test_csv = "/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/pets_test.csv" 
test_csv = "/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/peds_annotation_test_post_train.csv" 
df = pd.read_csv(test_csv)
print(f"Loaded {len(df)} test examples")


y_true = []
y_pred = []

for _, row in df.iterrows():
    file_path = row["file_path"]
    #label = int(row["adult_audio"])
    label = int(row["adult_audio"])

    # Load waveform
    waveform, orig_sr = sf.read(file_path)
    
    # Convert to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    
    # Resample to 16k
    if orig_sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=16000)
    
    # Feature extract
    inputs = feature_extractor(
        [waveform],
        sampling_rate=16000,
        return_tensors="pt",
        truncation=True,
        max_length=16000
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    
    y_true.append(label)
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Confusion Matrix

cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix ===")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["kid (0)", "adult (1)"]
)
disp.plot()
plt.show()


print("\n=== Classification Report ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=["kid(0)", "adult (1)"]
))


# F1 Scores

f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
f1_per_class = f1_score(y_true, y_pred, average=None)

print("\n=== F1 Scores ===")
print(f"Macro F1: {f1_macro:.4f}")
print(f"Micro F1: {f1_micro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")
print(f"Per-class F1: kid (0) = {f1_per_class[0]:.4f}, adult (1) = {f1_per_class[1]:.4f}")
