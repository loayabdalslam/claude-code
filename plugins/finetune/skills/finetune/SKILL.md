# Fine-tuning Skill

This skill provides a comprehensive guide and tools for fine-tuning Large Language Models (LLMs) using the Hugging Face ecosystem.

## Overview

Fine-tuning is the process of taking a pre-trained model and further training it on a specific dataset to adapt it for a particular task or domain. This plugin automates the standard workflow for Supervised Fine-Tuning (SFT).

## Workflow Steps

1.  **Data Ingestion**: Loading datasets from Hugging Face Hub or local CSV/JSON files.
2.  **Base Model Selection**: Choosing a pre-trained model (e.g., Gemma, Llama 3, Mistral).
3.  **Feature Engineering**: Applying instruction templates or formatting text fields.
4.  **Data Cleaning**: Removing noise and preparing a "Golden Dataset" for testing.
5.  **Parameter-Efficient Fine-Tuning (PEFT)**: Using LoRA/QLoRA to reduce VRAM requirements.
6.  **Training**: Executing the training loop with real-time TQDM progress bars.
7.  **Hub Integration**: Pushing the final weights or adapters back to Hugging Face.

## Essential Tools & Libraries

- **Transformers**: Core library for model loading and training.
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA/QLoRA).
- **TRL**: Transformer Reinforcement Learning (SFTTrainer).
- **BitsAndBytes**: 4-bit and 8-bit quantization for consumer GPUs.
- **Datasets**: Efficient data loading and mapping.

## Best Practices

- **Start Small**: Use the "Golden Dataset" feature to run a 5-minute training test before committing to a full run.
- **Quantization**: Always use 4-bit quantization (`load_in_4bit=True`) for large models unless you have high-end hardware.
- **Learning Rate**: For LoRA, a learning rate between `1e-4` and `3e-4` is usually optimal.
- **Monitoring**: Watch the training loss; if it doesn't decrease, check your data formatting.

## References

- [Hugging Face Documentation](https://huggingface.co/docs)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
