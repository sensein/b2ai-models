from datasets import load_dataset, Audio, Value
import pandas as pd
from transformers import AutoFeatureExtractor
import soundfile as sf
import evaluate
import numpy as np
from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from customTrainer.trainer import WeightedTrainer, calc_pos_weight_tensor
import librosa
import torch

# change this based on which files etc.

#dataset = load_dataset("csv", data_files="/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/peds_duration_19000.csv")
#dataset = load_dataset("csv", data_files="/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cat_dog.csv")

csv_files = {
    'train': '/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/ped_annotation_train_rebalance.csv', 
    'test': '/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/peds_annotations_test_20000.csv'
}

#dataset = load_dataset('csv', data_files=csv_files)
#dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42) # for pets

dataset = dataset.rename_column("file_path", "audio")
dataset = dataset.cast_column("audio", Value("string"))  # pyarrow thinks this is long string

# def decode_audio(example):
#     audio_array, sr = sf.read(example["audio"])
#     return {"audio": {"array": audio_array, "sampling_rate": sr, "path": example["audio"]}}

# dataset = dataset.map(decode_audio)
#dataset = dataset.shuffle(seed=42)

print(dataset)

label2id, id2label = dict(), dict()
id2label = {0: "kid", 1: "adult"}
label2id = {"adult": 1, "kid": 0}

# id2label = {0: "cat", 1: "dog"}
# label2id = {"dog": 1, "cat": 0}

num_labels = 2

# load transformer
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")

def preprocess_function(examples):
    audio_arrays = []
    for x in examples["audio"]:
        
        audio, orig_sr = sf.read(x)   # convert list → np.array
        
        if audio.ndim > 1:            # stereo
            audio = audio.mean(axis=1) # convert to mono
        #orig_sr = x["sampling_rate"]
        
        
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        audio_arrays.append(audio)
    
    #audio_arrays = [x["array"] for x in examples["audio"]]
    #audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_attention_mask=True, # attention masking to ignore padding during training
    #    max_length=16000,
    #    truncation=True,
    )
    return inputs

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
#encoded_dataset = encoded_dataset.rename_column("animal", "label")
encoded_dataset = encoded_dataset.rename_column("adult_audio", "label")

pos_weight = calc_pos_weight_tensor(encoded_dataset)

accuracy = evaluate.load("accuracy")

model = AutoModelForAudioClassification.from_pretrained(
    "microsoft/wavlm-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

for param in model.wavlm.parameters():
    param.requires_grad = False #True # tme to unfreeze 
    
for param in model.wavlm.encoder.layers[-3:].parameters():
    param.requires_grad = True


training_args = TrainingArguments(
    output_dir="./peds_voice_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    warmup_steps=0.1,
    logging_steps=10,
    report_to=["tensorboard"],
    logging_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    group_by_length=True, # to group longer audio segments together
)

trainer =  WeightedTrainer(
#trainer =  Trainer(
    model=model,
    class_weights=pos_weight.to(model.device),
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)


#inference
model = AutoModelForAudioClassification.from_pretrained("./pets_voice_model")
feature_extractor = AutoFeatureExtractor.from_pretrained("./pets_voice_model")
model.eval()  # Important: set to eval mode

test_audio = (
    #"/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cats_dogs/test/test/dog_barking_15.wav"
    # "/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cats_dogs/test/cats/cat_56.wav"
   # "/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cats_dogs/test/test/dog_barking_44.wav"
   #"/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cats_dogs/test/cats/cat_133.wav"
   #"/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/cats_dogs/test/test/dog_barking_44.wav"
   #"/orcd/data/satra/002/datasets/b2aivoice/post_3.0/data/peds/bids_w_features/sub-096sm/ses-218720d1-e753-4ec8-83eb-f31bc7e82b27/audio/sub-096sm_ses-218720d1-e753-4ec8-83eb-f31bc7e82b27_task-favorite-food-1.wav"
   "/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/test_audio/task-picture-37.wav"
)

#waveform, sr = librosa.load(test_audio, sr=16000)  # waveform is a 1D numpy array
 
waveform, orig_sr = sf.read(test_audio)   # convert list → np.array

if waveform.ndim > 1:            # stereo
    waveform = waveform.mean(axis=1) # convert to mono


if orig_sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=16000)

inputs = feature_extractor([waveform], sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()

print("Predicted label:", pred_label)
print("Probabilities:", probs)

# classifier = pipeline("audio-classification", model="./pet_voice_model")
# x = classifier(inputs['input_values'])
# print(x)
