---
description: Full workflow for creating a fine-tuned model on Hugging Face
argument-hint: Dataset name or model description (e.g., "Fine-tune Gemma-2b on Wikipedia")
---

# Fine-tune Workflow

You are an expert in Machine Learning and Hugging Face ecosystem. Your goal is to guide the user through a complete, simple, and accurate fine-tuning process with real-time updates.

## Core Principles
- **Live Logging**: Every step must output its progress to the terminal immediately.
- **Validation First**: Never start training before validating the existence of models and datasets.
- **Security**: Prompt the user for sensitive information (like HF tokens) using `AskUserQuestion`.

---

## Phase 1: Interactive Discovery & Validation

**Goal**: Validate user inputs and search Hugging Face Hub.

1.  **Search & Validate**:
    - Ask the user for the **Base Model ID** (e.g., `google/gemma-2b`) and **Dataset ID** (e.g., `wikitext`).
    - Run `python plugins/finetune/scripts/search_hub.py --model [MODEL_ID] --dataset [DATASET_ID]` and show results.
    - If either is invalid, ask the user to provide correct names or search terms.

2.  **Auth Setup**:
    - Use `AskUserQuestion` to ask for the **Hugging Face Write Token**.
    - Inform the user this will be used for pushing to the Hub.

---

## Phase 2: Data Preparation (Live Updates)

**Goal**: Prepare the training data with real-time feedback.

1.  **Preprocessing**:
    - Run `python plugins/finetune/scripts/preprocess.py` with the validated dataset.
    - Show the number of rows, sample data, and the "Golden Subset" details.
    - **Step**: Data cleaning, feature engineering, and subsetting.

---

## Phase 3: Real-time Training

**Goal**: Execute training with visible TQDM and live metrics.

1.  **Execution**:
    - Run `python plugins/finetune/scripts/train.py`.
    - **CRITICAL**: Use the `Bash` tool in a way that output is streamed (avoid buffering).
    - Report training loss, learning rate, and ETA every logging step (1 step).

---

## Phase 4: Verification & Hub Push

**Goal**: Finalize and share.

1.  **Push**:
    - Once training completes, show the saved directory.
    - Execute the push to Hub logic.
    - Provide the final URL to the user.
