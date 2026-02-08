import os
import argparse
import pandas as pd
from datasets import load_dataset, Dataset

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Basic cleaning: remove extra whitespace
    return " ".join(text.split())

def format_instruction(sample, instruction_template=None):
    """
    Apply an instruction template to a dataset sample.
    Example template: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    """
    if instruction_template:
        return instruction_template.format(**sample)
    return sample.get("text", "")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean dataset for fine-tuning")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name on HF or local path")
    parser.add_argument("--output_path", type=str, default="cleaned_dataset", help="Path to save the cleaned dataset")
    parser.add_argument("--text_column", type=str, default="text", help="Column containing text")
    parser.add_argument("--subset_size", type=int, default=None, help="Create a small subset (Golden Dataset)")
    parser.add_argument("--template", type=str, help="Instruction template")

    args = parser.parse_args()

    print(f"--- Loading Dataset: {args.dataset_name} ---")
    try:
        if os.path.exists(args.dataset_name):
            if args.dataset_name.endswith(".csv"):
                df = pd.read_csv(args.dataset_name)
                dataset = Dataset.from_pandas(df)
            elif args.dataset_name.endswith(".json") or args.dataset_name.endswith(".jsonl"):
                df = pd.read_json(args.dataset_name, lines=True)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = load_dataset("text", data_files=args.dataset_name)["train"]
        else:
            dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"--- Cleaning Data ---")
    def process_func(example):
        example[args.text_column] = clean_text(example[args.text_column])
        if args.template:
            example["formatted_text"] = format_instruction(example, args.template)
        return example

    dataset = dataset.map(process_func)

    if args.subset_size:
        print(f"--- Creating Golden Subset (size={args.subset_size}) ---")
        dataset = dataset.select(range(min(len(dataset), args.subset_size)))

    print(f"--- Saving Processed Dataset to {args.output_path} ---")
    dataset.save_to_disk(args.output_path)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
