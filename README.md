# Claude Code Plugins Marketplace

Welcome to the official marketplace for Claude Code plugins. This repository serves as a hub for specialized tools and agents that extend the capabilities of Claude Code.

## 🚀 Featured Plugin: Finetune

The **Finetune** plugin is a professional-grade workflow for fine-tuning Large Language Models (LLMs) on Hugging Face datasets.

### Key Features:
- **Zero-Config PEFT**: Automatically detects model architectures for LoRA.
- **Real-time Monitoring**: Live TQDM progress bars and metric logging in your terminal.
- **Secure Integration**: Guided Hugging Face Hub uploads with secure token handling.
- **Data Engineering**: Built-in cleaning, formatting, and "Golden Subset" generation.

## 📦 How to Install Plugins

1. **Clone the Marketplace**:
   ```bash
   git clone https://github.com/loaiabdalslam/claude-code-plugins.git
   ```

2. **Install Dependencies**:
   Navigate to the specific plugin directory and install the requirements:
   ```bash
   pip install -r plugins/finetune/requirements.txt
   ```

3. **Run Claude Code**:
   Simply run `claude` in the root directory. Claude will automatically detect the plugins in the `plugins/` folder.

## 🛠️ Usage

Invoke the featured plugin using its slash command:

```text
/finetune "Fine-tune Gemma-2b on Wikipedia"
```

---
Built with ❤️ by **loaiabdalslam**
