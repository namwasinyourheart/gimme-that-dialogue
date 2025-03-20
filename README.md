# 📖 Gimme That Dialogue: Finetuning LLMs for Dialogue Summarization
## 📖 Overview
This project focuses on fine-tuning Large Language Models (LLMs) such as T5, LLaMA, and Qwen to enhance dialogue summarization. The goal is to improve the conciseness, coherence, and informativeness of generated summaries by training the models on domain-specific datasets.

## ✨ Highlights
- Fine-tuning transformer-based LLMs for dialogue summarization
- Comparison of different models (BART, T5, LLaMA, Qwen, etc.)
- Evaluating improvements in summary quality before and after training
- Using various evaluation metrics such as ROUGE, BLEU, and human qualitative analysis
- Optimized training with LoRA/QLoRA and efficient dataset preprocessing
## 📈 Results
  
| Experiment                        | ROUGE-1  | ROUGE-2  | ROUGE-L  | ROUGE-Lsum |
|-----------------------------------|----------|----------|----------|------------|
| bart-large-cnn__samsum            | 0.2447   | 0.0912   | 0.1855   | 0.1858     |
| qwen2.5-1.5b-instruct__samsum     | 0.1976   | 0.0563   | 0.1406   | 0.1495     |
| llama-3.2-1b-instruct__samsum      | 0.2560   | 0.0834   | 0.1934   | 0.1933     |
| llama-3.2-3b-instruct__samsum      | 0.2890   | 0.1047   | 0.2193   | 0.2198     |
| t5-small__samsum                  | 0.2261   | 0.0623   | 0.1751   | 0.1750     |
| flan-t5-base__samsum              | 0.4733   | 0.2301   | 0.3958   | 0.3957     |
  