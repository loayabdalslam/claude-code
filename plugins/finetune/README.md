# Hugging Face Fine-tuning Plugin

This plugin automates the process of fine-tuning Large Language Models (LLMs) on Hugging Face datasets. It provides a structured workflow from data ingestion to model deployment.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r plugins/finetune/requirements.txt
```

## How to Use

Invoke the fine-tuning workflow using the slash command:

```
/finetune "Fine-tune [Model] on [Dataset]"
```

Example: `/finetune "Fine-tune Gemma-2b on Wikipedia"`

## Plugin Components

- **`/finetune` Command**: A guided workflow that takes you through data preparation, cleaning, training, and pushing to the Hub.
- **`finetune-expert` Agent**: A specialized AI agent that can help you debug training scripts and optimize hyperparameters.
- **`scripts/preprocess.py`**: Handles data loading, cleaning, and creation of a "Golden Dataset" (subset) for testing.
- **`scripts/train.py`**: A robust training script using `transformers`, `peft` (QLoRA), and `trl` (SFTTrainer) with real-time TQDM progress.

## Workflow Steps

1.  **Import Data**: Load from HF Hub or local CSV/JSON.
2.  **Import Base Model**: Select your foundation model.
3.  **Feature Engineering**: Apply instruction templates.
4.  **Data Cleaning**: Remove noise and outliers.
5.  **Tests**: Generate a small "Golden Subset" for validation.
6.  **Write Code**: The system generates a tailored `train.py`.
7.  **Run & Monitor**: Execute training with real-time terminal updates.
8.  **Push to Hub**: Automatically upload results to Hugging Face.

## Workspace

All logs, results, and saved models are stored in `plugins/finetune/workspace/`.
