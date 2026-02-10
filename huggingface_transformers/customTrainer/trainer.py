from transformers import Trainer
import torch.nn as nn
import torch


def calc_pos_weight_tensor(dataset):
    num_pos = sum(dataset["train"]["label"])
    num_neg = len(dataset["train"]["label"]) - num_pos
    pos_weight = num_neg/num_pos
    neg_weight = 1.0
    weight_tensor = torch.tensor([neg_weight, pos_weight])
    return weight_tensor


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) :
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
