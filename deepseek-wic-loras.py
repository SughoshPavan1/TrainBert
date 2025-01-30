from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import os

# 🔹 Set experiment name (change this every time to avoid AzureML errors)
EXPERIMENT_NAME = "wic_experiment_v2"  # Change version each time if needed

# 🔹 Detect device (force CPU)
device = "cpu"

# 🔹 Use DeepSeek Coder 1.3B (smallest DeepSeek model that can run on CPU)
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
print(f"Using model: {model_name} on {device}")

# 🔹 Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 Load model with float32 (since float16 is not supported on CPU)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification (WiC dataset)
    torch_dtype=torch.float32  # Must use float32 on CPU
).to(device)

# 🔹 Load dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset("super_glue", "wic")

def preprocess_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 🔹 Define training arguments (shortened for AzureML compatibility)
training_args = TrainingArguments(
    output_dir=f"./wic_model_{EXPERIMENT_NAME}",  # Shortened directory
    num_train_epochs=3,  # Reduce if needed
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",  # Shortened to avoid AzureML errors
    save_strategy="epoch",  # Shortened to avoid AzureML errors
    save_steps=500,
    load_best_model_at_end=True,
    report_to=[],
    metric_for_best_model="accuracy",  # Shortened to avoid AzureML errors
)

# 🔹 Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# 🔹 Clear MLflow cache (fix for AzureML parameter errors)
print("Clearing MLflow cache to prevent parameter conflicts...")
os.system("rm -rf mlruns")  # Clears MLflow cache

# 🔹 Start Training
print("Starting training on CPU (this may take a long time)...")
trainer.train()

# 🔹 Save Final Model
model.save_pretrained(f"./wic_model_{EXPERIMENT_NAME}")
print(f"Training completed. Model saved to './wic_model_{EXPERIMENT_NAME}'.")
