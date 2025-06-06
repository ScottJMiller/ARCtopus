# src/training/train_codegemma.py

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Imports for fine-tuning
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import numpy as np

from datasets import load_dataset

# Adjust sys.path to find src.common and src.orchestrator
# Vertex AI training environment will usually place your package's root on sys.path
# but explicitly adding it here for robustness if run standalone.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common import load_arc_challenges, grid_to_text, text_to_grid, COLOR_NAMES
from src.tentacles.program_synthesis_tentacle import ProgramSynthesisTentacle # To load the model

# --- Data Preparation for Fine-tuning ---
class ARCFinetuningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, tokenizer, max_length: int = 512):
        """
        Loads fine-tuning data from a Hugging Face dataset and tokenizes it.

        Args:
            dataset_name (str): The Hugging Face dataset ID (e.g., "barc0/arc-agi-data-prompt-formatted-train").
            tokenizer: The Hugging Face tokenizer.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the dataset from Hugging Face
        # This will download the dataset to your Paperspace/Colab environment.
        print(f"[ARCFinetuningDataset] Loading dataset: {dataset_name}")
        raw_dataset = load_dataset(dataset_name, split="train") # Assuming "train" split
        #print(f"[ARCFinetuningDataset] Dataset loaded with {len(raw_dataset)} examples.")

        self.data = []
        for raw_item in raw_dataset:
            item: Dict[str, Any] = dict(raw_item) # Explicitly cast to dict
            prompt = item['prompt']
            completion = item['completion'] # This is the Python code solution

            # Combine prompt and completion for training
            # The model learns to generate 'completion' given 'prompt'
            # Ensure it mirrors the inference prompt structure (e.g., ends with ````python` and `def transform(...)`)
            # and then continues with the completion.

            # The prompt from the HF dataset already seems to include the common library functions.
            # So we just concatenate prompt and completion.
            full_text = prompt + completion # Concatenate prompt and the correct code solution

            self.data.append(full_text)

        # Add a print here to show the size of the *processed* dataset
        print(f"[ARCFinetuningDataset] Processed {len(self.data)} examples for fine-tuning.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the full text
        encoding = self.tokenizer(
            self.data[idx],
            max_length=self.max_length,
            padding="max_length", # Pad to max_length
            truncation=True,     # Truncate if longer than max_length
            return_tensors="pt"
        )

        # For causal language models, labels are usually the input_ids shifted by one.
        # However, `Trainer` often handles this shifting internally if labels=input_ids.
        # We explicitly clone to ensure labels are separate tensors.
        #labels = encoding["input_ids"].squeeze().clone()

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].clone() # Keep clone for labels
        }


# --- Main Training Function ---
def train_model(
    model_name: str,
    dataset_name: str, # New parameter for dataset ID
    data_dir: str,      # Still kept, but now used for local ARC JSON if needed
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    load_in_4bit: bool = True
):
    # Load tokenizer and model (similar to Tentacle's __init__)
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token_id is set for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Common practice

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if load_in_4bit else None,
        torch_dtype=torch.bfloat16 if load_in_4bit else torch.float16,
        device_map="auto" if device == "cuda" else None # Let Hugging Face manage device placement
    )
    model.train() # Set model to training mode

    # --- Data Loading ---
    # Old arc_data_path and challenges loading is now replaced by Hugging Face dataset loading
    # arc_data_path = Path(data_dir) / "arc-agi_training_challenges.json"
    # challenges = load_arc_challenges(str(arc_data_path))

    # Instantiate the new dataset using dataset_name
    train_dataset = ARCFinetuningDataset(dataset_name, tokenizer)
    
    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1, # Save only the last checkpoint
        report_to="none", # Disable integrations like Weights & Biases for simplicity
        fp16=True, # Use mixed precision if not 4-bit (bfloat16 is usually on by default for 4-bit)
        bf16=False, # Set bf16 to true if using 4-bit (or target bf16 capable GPUs)
        gradient_checkpointing=True
        # Add evaluation strategy if you have a validation set
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer, # Pass tokenizer to Trainer for input handling
        # data_collator=data_collator, # Might need a custom data collator for variable length sequences
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeGemma on ARC-like data.")
    parser.add_argument("--model_name", type=str, default="google/codegemma-2b", help="Hugging Face model ID.")
    parser.add_argument("--dataset_name", type=str, default="barc0/arc-agi-data-prompt-formatted-train", help="Hugging Face dataset ID.") # NEW ARG
    parser.add_argument("--data_dir", type=str, default="data/arc-prize-2025", help="Local or GCS path to ARC data (less critical now).") # Keep if needed for other parts
    parser.add_argument("--output_dir", type=str, default="models/codegemma_finetuned", help="Local or GCS path to save model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization.")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name, # Pass the new dataset name
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        load_in_4bit=args.load_in_4bit
    )