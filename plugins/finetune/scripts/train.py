import os
import torch
import argparse
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on Hugging Face")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name on HF")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name on HF")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="Field in dataset containing text")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for pushing to hub")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to HF hub")
    parser.add_argument("--repo_id", type=str, help="Repo ID for HF hub")

    args = parser.parse_args()

    print(f"--- Loading Dataset: {args.dataset_name} ---")
    dataset = load_dataset(args.dataset_name, split="train")

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"--- Loading Model: {args.model_name} ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # PEFT configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common for Llama/Gemma
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"--- Training Configuration ---")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=1, # Real-time logging
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs",
        push_to_hub=args.push_to_hub,
        hub_token=args.hf_token,
        hub_model_id=args.repo_id if args.repo_id else None,
        disable_tqdm=False # Ensure tqdm is active
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    print(f"--- Starting Training ---")
    sys.stdout.flush()
    trainer.train()

    print(f"--- Saving Model ---")
    trainer.model.save_pretrained(args.output_dir)

    if args.push_to_hub and args.repo_id:
        print(f"--- Pushing to Hub: {args.repo_id} ---")
        trainer.push_to_hub()

if __name__ == "__main__":
    main()
