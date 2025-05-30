# 🧬 Protein Finetuning Pipeline

A scalable, modular deep learning pipeline for finetuning ESM-2 using:
- 🧠 Multi-task support (regression, classification, multi-label, multiclass)
- ⚡ Ray + Lightning + FSDP for distributed training
- 🔄 LoRA integration for low-rank fine-tuning
- 📊 MLflow tracking
- 🐳 Docker-ready for deployment

## 🐳 Docker Support
Build: docker build -t protein-trainer .
Run: docker run --rm protein-trainer train-debug --help

## ⚙️ CLI Launcher
python run_train_cli.py run --mode debug --task-type regression
