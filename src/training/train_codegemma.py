# src/training/train_codegemma.py

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
import time

# Imports for fine-tuning
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import numpy as np

from huggingface_hub import whoami, login
from datasets import load_dataset

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
        print(f"[ARCFinetuningDataset] Loading dataset: {dataset_name}")
        raw_dataset = load_dataset(dataset_name, split="train") # Assuming "train" split

        self.data = []
        for raw_item in raw_dataset:
            item: Dict[str, Any] = dict(raw_item)
            prompt = item.get('prompt', '')
            completion = item.get('completion', '')

            # Ensure prompt and completion are strings
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
                
            # Combine prompt and completion for training
            # The model learns to generate 'completion' given 'prompt'
            full_text = prompt + completion
            self.data.append(full_text)

        print(f"[ARCFinetuningDataset] Processed {len(self.data)} examples for fine-tuning.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the full text
        encoding = self.tokenizer(
            self.data[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # The Trainer will automatically handle shifting the labels for causal language modeling.
        # We just need to provide the input_ids as labels.
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone() # Labels are the same as input_ids for CLM
        }


# --- Main Training Function ---
def train_model(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    load_in_4bit: bool = True
):
    # --- Model and Tokenizer Setup ---
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
    if tokenizer.pad_token is None:
        # Set pad token to eos token if not set
        tokenizer.pad_token = tokenizer.eos_token
        
    # The main error occurs here. The combination of arguments, especially
    # torch_dtype alongside a quantization_config that specifies its own dtype,
    # can cause conflicts in some versions of the library.
    # The traceback's "TypeError: argument of type 'NoneType' is not iterable"
    # suggests a model configuration value is not being set correctly upon initialization.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # Use "auto" to leverage accelerate for device placement
        # torch_dtype removed to avoid conflict with bnb_config's compute_dtype
    )

    # --- Dataset and Training Arguments ---
    train_dataset = ARCFinetuningDataset(dataset_name, tokenizer)
    
    # --- Training Arguments Correction ---
    # 1. bf16=True is set for consistency, as the bnb_config uses bfloat16.
    #    This is generally better than fp16 on compatible hardware (Ampere+).
    # 2. fp16 is set to False to avoid conflicts with bf16.
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
        save_total_limit=1,
        report_to="none",
        bf16=True, # CORRECTED: Use bf16 for consistency with compute_dtype
        fp16=False, # CORRECTED: Ensure fp16 is disabled when bf16 is active
        gradient_checkpointing=True,
    )

    # --- Trainer Correction ---
    # The 'processing_class' argument is not a valid argument for the Trainer.
    # The Trainer automatically handles tokenization when a tokenizer is available
    # and the dataset returns tokenized inputs.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # --- Save Final Model ---
    # Use the output_dir directly, as it's the final save location.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    # It's recommended to handle tokens securely, e.g., via environment variables
    # or notebook secrets, rather than hardcoding them.
    try:
        # Use HUGGING_FACE_HUB_TOKEN environment variable if available
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "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")
        whoami(token=token)
        login(token=token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}")
        print("Proceeding without login. Access to private models/datasets will be restricted.")

    parser = argparse.ArgumentParser(description="Fine-tune CodeGemma on ARC-like data.")
    parser.add_argument("--model_name", type=str, default="google/codegemma-2b", help="Hugging Face model ID.")
    parser.add_argument("--dataset_name", type=str, default="barc0/arc-agi-data-prompt-formatted-train", help="Hugging Face dataset ID.")
    parser.add_argument("--output_dir", type=str, default="./models/codegemma_finetuned", help="Local path to save model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization.")
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit", help="Disable 4-bit quantization.")

    # In a script, you might want to parse known args if running in an environment
    # like Jupyter that adds its own arguments.
    args, unknown = parser.parse_known_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        load_in_4bit=args.load_in_4bit
    )
