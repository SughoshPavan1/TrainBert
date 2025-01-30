%pip install transformers[torch] datasets evaluate torch

!pip freeze > requirements.txt

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
import evaluate
import numpy as np
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load WiC dataset
dataset = load_dataset("super_glue", "wic")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], 
        examples["sentence2"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Custom callback for progress tracking
class CustomTrainingCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = []
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            if state.log_history:
                loss = state.log_history[-1].get('loss', 'N/A')
                print(f"Step: {state.global_step}, Loss: {loss}")

# Training arguments with MLflow completely disabled
training_args = TrainingArguments(
    output_dir="./wic_bert_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[],  # Disable all reporting integrations
    disable_tqdm=False
)

# Create trainer with minimal setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[CustomTrainingCallback()]
)

# Train and evaluate
print("Starting training...")
trainer.train()
print("Training completed!")

final_metrics = trainer.evaluate()
print(f"Final evaluation metrics: {final_metrics}")

# Save the model
model_path = "./wic_bert_model"
trainer.save_model(model_path)
print(f"Model saved to: {model_path}")

# Register model in Azure ML (with error handling)
try:
    from azureml.core import Workspace, Model
    ws = Workspace.from_config()
    Model.register(
        workspace=ws,
        model_path=model_path,
        model_name="wic_bert_model",
        tags={
            "task": "text_classification",
            "dataset": "WiC",
            "accuracy": str(final_metrics["eval_accuracy"])
        }
    )
    print("Model registered successfully in Azure ML")
except Exception as e:
    print(f"Error registering model: {str(e)}")
    print("You can still find the saved model in:", model_path)
