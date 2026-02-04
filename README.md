# VLM Fine-tuning Pipeline for Qwen3-VL-30B using Tinker

Complete pipeline for fine-tuning Qwen3-VL-30B vision-language model on forest fire detection using the D-Fire dataset and Tinker API.

## Overview

This pipeline handles:

- Loading and preprocessing image-text training data
- Converting to Tinker VLM format
- Uploading dataset to Tinker
- Submitting fine-tuning job

## Prerequisites

1. **API Keys**: You need both Tinker and HuggingFace API keys
2. **Python**: Python 3.11 or higher
3. **Dataset**: Training data in JSONL format with corresponding images

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Eashwar-S/vlm_finetuning.git
cd vlm_finetuning
```

### 2. Create Virtual Environment

```bash
python -m venv vlm_venv
# On Windows:
vlm_venv\Scripts\activate
# On Linux/Mac:
source vlm_venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# On Windows:
echo TINKER_API_KEY=your_tinker_api_key_here > .env
echo OPENROUTER_API_KEY=your_openrouter_api_key_here >> .env
echo WANDB_API_KEY=your_wandb_api_key_here >> .env
echo HUGGINGFACE_API_KEY=your_huggingface_api_key_here >> .env

# On Linux/Mac:
cat > .env << EOF
TINKER_API_KEY=your_tinker_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
EOF
```

**Important:** Replace `your_tinker_api_key_here` and `your_huggingface_api_key_here` with your actual API keys.

### 5. Prepare D-Fire Dataset

This project uses the D-Fire dataset for forest fire detection. Ensure your dataset is organized as:

```
dataset/
├── data/
│   ├── train.jsonl              # Full training dataset
│   ├── train_short.jsonl        # Subset for quick testing
│   └── train_short_formatted.jsonl  # Formatted for Tinker (auto-generated)
├── images/                      # Full D-Fire image dataset (not in git)
└── images_short/                # Subset of images like 100 (not in git)
```

> [!NOTE]
> The `images/` and `images_short/` directories are excluded from Git via `.gitignore` due to their size.

## Usage

### Step 1: Run the Fine-tuning Pipeline

```bash
python train_vlm_simple.py
```

This script will:

1. Load your training data from `dataset/data/train_short.jsonl`
2. Load corresponding images from `dataset/images_short/`
3. Convert images to base64 and format data for Tinker VLM API
4. Upload the dataset to Tinker
5. Create and submit a fine-tuning job
6. Monitor training progress and save checkpoints to `experiments/`

**Expected Output:**

```
================================================================================
VLM Fine-tuning Pipeline for Qwen3-VL-30B
================================================================================

[Step 1/4] Loading and preparing training data...
Loading data from dataset/data/train_short.jsonl...
Processed 10 examples...
...
Successfully loaded 103 training examples

[Step 2/4] Saving formatted dataset...
Saved formatted dataset to dataset/data/train_short_formatted.jsonl

[Step 3/4] Uploading dataset to Tinker...
Dataset uploaded successfully! Dataset ID: ds_xxxxx

[Step 4/4] Creating fine-tuning job...
Fine-tuning job created successfully! Job ID: ft_xxxxx

================================================================================
Fine-tuning job submitted successfully!
================================================================================

Job ID: ft_xxxxx
Dataset ID: ds_xxxxx
Model: Qwen/Qwen3-VL-30B-A3B-Instruct
```

### Step 2: Export the Trained Model

After training completes, export the checkpoint for inference:

```bash
python export_model.py
```

This will export the model from `experiments/` to `exported_models/` in a format compatible with vLLM.

### Step 3: Run Inference

Use the exported model for fast inference with vLLM:

```bash
python inference_vllm.py --model-path exported_models/your_model --image path/to/test_image.jpg
```

## Data Format

### Input JSONL Format

Each line in your training JSONL should contain:

```json
{
  "image": "WEB10440.jpg",
  "filename": "WEB10440.jpg",
  "prompt": "Analyze the aerial image from the UAV's camera and detect potential forest fires...",
  "response_schema": {...},
  "teacher_answer": {
    "forest_fire_smoke_visible": "Yes",
    "forest_fire_flames_visible": "Yes",
    ...
  }
}
```

### Tinker VLM Format (Auto-converted)

The script automatically converts to Tinker's format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "data:image/jpeg;base64,..."
        },
        {
          "type": "text",
          "text": "Your prompt here"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"forest_fire_smoke_visible\": \"Yes\", ...}"
    }
  ]
}
```

## Hyperparameters

Default hyperparameters (can be modified in `vlm_finetune_tinker.py`):

```python
hyperparameters = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_length": 2048
}
```

### Hyperparameter Tuning Guide

- **learning_rate**: Start with 2e-5, decrease if training is unstable
- **num_train_epochs**: 3-5 epochs for most cases
- **per_device_train_batch_size**: Keep at 1 for large VLMs
- **gradient_accumulation_steps**: Increase to simulate larger batch sizes
- **lora_r**: Higher values (16-32) for more complex tasks
- **lora_alpha**: Typically 2x the lora_r value

## Troubleshooting

### API Key Issues

```
ValueError: TINKER_API_KEY not found in environment variables
```

**Solution**: Ensure your `.env` file is in the project root and contains valid API keys.

### Image Not Found

```
Warning: Image not found: dataset/images_short/WEB10440.jpg
```

**Solution**: Verify that image filenames in JSONL match actual files in the images directory.

### Upload Failures

```
Failed to upload dataset: 413 - Payload Too Large
```

**Solution**: The dataset might be too large. Consider:

- Reducing image resolution
- Splitting into smaller batches
- Using fewer training examples

### Memory Issues

**Solution**:

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

## File Structure

```
vlm_finetuning/
├── train_vlm_simple.py        # Main training script using Tinker
├── dataset_creation.py        # Create D-Fire dataset with API calls
├── export_model.py            # Export Tinker checkpoints for vLLM
├── inference_vllm.py          # Fast inference with vLLM
├── finetune.ipynb             # Jupyter notebook for interactive training
├── requirements.txt           # Python dependencies (consolidated)
├── .env                       # API keys (not in git)
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── GITHUB_SETUP.md            # GitHub repository setup guide
├── dataset/
│   ├── data/
│   │   ├── train.jsonl                    # Full D-Fire training data
│   │   ├── train_short.jsonl              # Subset for testing
│   │   └── train_short_formatted.jsonl    # Tinker-formatted (auto-generated)
│   ├── images/                # Full D-Fire image dataset (not in git)
│   └── images_short/          # Image subset (not in git)
├── experiments/               # Training outputs and checkpoints (not in git)
└── exported_models/           # vLLM-compatible models (not in git)
```

## API Reference

### TinkerVLMFineTuner Class

#### Methods

- `load_and_prepare_data(jsonl_path, images_dir)`: Load and format training data
- `upload_dataset(dataset_path, dataset_name)`: Upload dataset to Tinker
- `create_fine_tuning_job(dataset_id, model_name, job_name, hyperparameters)`: Create fine-tuning job
- `check_job_status(job_id)`: Check job status
- `list_fine_tuned_models()`: List all fine-tuned models

## Best Practices

1. **Data Quality**: Ensure high-quality, diverse training examples
2. **Image Size**: Optimize images to balance quality and file size
3. **Validation Set**: Keep a separate validation set for evaluation
4. **Monitoring**: Regularly check training progress
5. **Versioning**: Track dataset versions and hyperparameters

## Resources

- [Tinker API Documentation](https://tinker-docs.thinkingmachines.ai)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)

## Support

For issues or questions:

1. Check Tinker API documentation
2. Review error messages and logs
3. Verify API key permissions
4. Contact Tinker support

## License

This pipeline is provided as-is for use with Tinker API.
