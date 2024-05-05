""""
Trains a DeIT model on the DocOrNot dataset to classify images as documents or pictures.
"""
import os

from transformers import (
    DeiTFeatureExtractor,
    DeiTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback,
)
from codecarbon import track_emissions
import numpy as np
import evaluate
from datasets import load_dataset
import torch
from transformers.trainer_callback import EarlyStoppingCallback


BASE_MODEL = "facebook/deit-tiny-distilled-patch16-224"
# facebook/deit-base-distilled-patch16-224"
# facebook/deit-tiny-patch16-224
# facebook/deit-small-distilled-patch16-224
# facebook/deit-tiny-distilled-patch16-224


def best_device():
    # picking the best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (Nvidia GPU).")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.file_path, "a") as f:
                f.write(f"{metrics}\n")


metrics_logger_callback = MetricsLoggerCallback(
    os.path.join(os.path.dirname(__file__), "metrics.txt")
)


accuracy = evaluate.load("accuracy")

feature_extractor = DeiTFeatureExtractor.from_pretrained(BASE_MODEL)


def preprocess_images(examples):
    examples["pixel_values"] = feature_extractor(
        examples["image"], return_tensors="pt"
    ).pixel_values

    return examples


data_collator = DefaultDataCollator()


label2id = {"picture": 0, "document": 1}
id2label = {0: "picture", 1: "document"}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # one epoch seems good enough
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


@track_emissions(project_name="DocOrNot")
def train():
    dataset = load_dataset("mozilla/docornot")
    dataset = dataset.rename_column("is_document", "label")
    prepared_dataset = dataset.map(
        preprocess_images, batched=True, remove_columns=["image"]
    )

    model = DeiTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(best_device())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["test"],
        data_collator=DefaultDataCollator(),
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            metrics_logger_callback,
        ],
    )

    trainer.train()
    trainer.push_to_hub("mozilla/docornot")


if __name__ == "__main__":
    train()
