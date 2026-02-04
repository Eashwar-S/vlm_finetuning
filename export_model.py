"""
Export trained LoRA model from Tinker checkpoint for vLLM inference
This script downloads the checkpoint from Tinker storage and saves it as a LoRA adapter
"""

import os
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
import argparse
import urllib.request

import torch
from dotenv import load_dotenv

# Load environment variables (for TINKER_API_KEY)
load_dotenv()

# Tinker imports
import tinker
from tinker_cookbook.checkpoint_utils import load_checkpoints_file, get_last_checkpoint


def export_checkpoint_sync(
    experiment_dir: str,
    checkpoint_name: Optional[str] = None,
    output_dir: str = "./exported_models",
):
    """
    Export a Tinker checkpoint to LoRA adapter format for vLLM
    
    Args:
        experiment_dir: Path to experiment directory (e.g., "./experiments/forest_fire_vlm")
        checkpoint_name: Name of checkpoint to export (e.g., "000200"). If None, uses last checkpoint.
        output_dir: Where to save the exported model
    """
    
    experiment_path = Path(experiment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EXPORTING TINKER CHECKPOINT")
    print("=" * 80)
    
    # Load checkpoint metadata
    if checkpoint_name is None:
        checkpoint_info = get_last_checkpoint(str(experiment_path))
        if checkpoint_info is None:
            raise ValueError(f"No checkpoints found in {experiment_path}")
        print(f"\nUsing last checkpoint: {checkpoint_info['name']}")
    else:
        checkpoints = load_checkpoints_file(str(experiment_path))
        checkpoint_info = None
        for ckpt in checkpoints:
            if ckpt['name'] == checkpoint_name:
                checkpoint_info = ckpt
                break
        
        if checkpoint_info is None:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not found")
    
    print(f"\nCheckpoint Info:")
    print(f"  Name: {checkpoint_info['name']}")
    print(f"  Epoch: {checkpoint_info.get('epoch', 'N/A')}")
    print(f"  Batch: {checkpoint_info.get('batch', 'N/A')}")
    print(f"  State Path: {checkpoint_info.get('state_path', 'N/A')}")
    print(f"  Sampler Path: {checkpoint_info.get('sampler_path', 'N/A')}")
    
    # Load config to get model name
    config_file = experiment_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.loads(f.read())
    
    base_model_name = config['model_name']
    lora_rank = config.get('lora_rank', 16)
    
    print(f"\nBase Model: {base_model_name}")
    print(f"LoRA Rank: {lora_rank}")
    
    # For vLLM, we need the sampler weights (inference weights)
    sampler_path = checkpoint_info.get('sampler_path')
    if not sampler_path:
        print("\n⚠️  Warning: No sampler_path found. Using state_path instead.")
        sampler_path = checkpoint_info.get('state_path')
    
    if not sampler_path:
        raise ValueError("No checkpoint path found (neither sampler_path nor state_path)")
    
    print(f"\nDownloading checkpoint from: {sampler_path}")
    
    # Use official Tinker API to download checkpoint
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    future = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path)
    checkpoint_archive_url_response = future.result()
    
    print(f"Got signed URL (expires: {checkpoint_archive_url_response.expires})")
    
    # Download the checkpoint archive
    with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        print(f"Downloading checkpoint archive...")
        urllib.request.urlretrieve(checkpoint_archive_url_response.url, tmp_path)
        print(f"✓ Downloaded to temporary file")
    
    # Extract the archive and load the checkpoint
    print("Extracting checkpoint...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(tmp_path, 'r') as tar:
            tar.extractall(tmp_dir)
        
        # Find the checkpoint file (safetensors or pytorch format)
        extracted_files = list(Path(tmp_dir).rglob('*'))
        checkpoint_file = None
        
        # Prefer safetensors format
        for f in extracted_files:
            if f.is_file() and f.suffix == '.safetensors':
                checkpoint_file = f
                break
        
        # Fall back to pytorch format
        if checkpoint_file is None:
            for f in extracted_files:
                if f.is_file() and f.suffix in ['.pt', '.pth', '.bin'] and f.stat().st_size > 1000:
                    checkpoint_file = f
                    break
        
        # Last resort: largest file
        if checkpoint_file is None:
            checkpoint_file = max(extracted_files, key=lambda f: f.stat().st_size if f.is_file() else 0)
        
        print(f"Loading checkpoint from: {checkpoint_file.name}")
        
        # Load based on file format
        if checkpoint_file.suffix == '.safetensors':
            # Use safetensors library
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_file)
                print("✓ Loaded safetensors checkpoint")
            except ImportError:
                print("⚠️  safetensors not installed, installing...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'safetensors'])
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_file)
                print("✓ Loaded safetensors checkpoint")
        else:
            # Use PyTorch with weights_only=False for compatibility
            state_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            print("✓ Loaded PyTorch checkpoint")
    
    # Clean up temporary tar file
    os.unlink(tmp_path)
    
    # Save the LoRA adapter weights
    save_path = output_path / f"checkpoint_{checkpoint_info['name']}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving LoRA adapter to: {save_path}")
    
    # Save the state dict
    torch.save(state_dict, save_path / "adapter_model.bin")
    
    # Create adapter config for vLLM/PEFT
    adapter_config = {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": lora_rank * 2,  # Common default
        "lora_dropout": 0.0,  # Inference mode
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "revision": None,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Common VLM targets
        "task_type": "CAUSAL_LM"
    }
    
    with open(save_path / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    # Save metadata
    metadata = {
        "base_model": base_model_name,
        "checkpoint_name": checkpoint_info['name'],
        "epoch": checkpoint_info.get('epoch'),
        "batch": checkpoint_info.get('batch'),
        "lora_rank": lora_rank,
        "tinker_path": sampler_path,
        "export_type": "lora_adapter",
    }
    
    with open(save_path / "export_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print(f"\nLoRA adapter saved to: {save_path}")
    print(f"\nTo use with vLLM:")
    print(f"  python inference_vllm.py \\")
    print(f"    --model_path {save_path} \\")
    print(f"    --use-lora \\")
    print(f"    --base-model {base_model_name} \\")
    print(f"    --image_dir ./dataset/images_short \\")
    print(f"    --output predictions.jsonl")
    
    return save_path


def export_checkpoint(
    experiment_dir: str,
    checkpoint_name: Optional[str] = None,
    output_dir: str = "./exported_models",
):
    """Export checkpoint (now synchronous)"""
    return export_checkpoint_sync(
        experiment_dir=experiment_dir,
        checkpoint_name=checkpoint_name,
        output_dir=output_dir,
    )


def list_checkpoints(experiment_dir: str):
    """List all available checkpoints"""
    checkpoints = load_checkpoints_file(experiment_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {experiment_dir}")
        return
    
    print("\nAvailable Checkpoints:")
    print("-" * 80)
    print(f"{'Name':<12} {'Epoch':<8} {'Batch':<8} {'Has State':<12} {'Has Sampler'}")
    print("-" * 80)
    
    for ckpt in checkpoints:
        has_state = '✓' if 'state_path' in ckpt else '✗'
        has_sampler = '✓' if 'sampler_path' in ckpt else '✗'
        epoch = ckpt.get('epoch', 'N/A')
        batch = ckpt.get('batch', 'N/A')
        print(f"{ckpt['name']:<12} {epoch:<8} {batch:<8} {has_state:<12} {has_sampler}")


def main():
    parser = argparse.ArgumentParser(description="Export Tinker checkpoint for vLLM inference")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="./experiments/forest_fire_vlm",
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name to export (e.g., '000200'). If not provided, uses last checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exported_models",
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints and exit"
    )
    
    args = parser.parse_args()
    
    # If --list flag, just list checkpoints
    if args.list:
        list_checkpoints(args.experiment_dir)
        print("\nTo export a checkpoint, run:")
        print(f"  python export_model.py --checkpoint <name>")
        return
    
    # If no checkpoint specified, list them first
    if args.checkpoint is None:
        print("No checkpoint specified. Available checkpoints:")
        list_checkpoints(args.experiment_dir)
        print("\nExporting last checkpoint...")
    
    # Export the checkpoint
    export_checkpoint(
        experiment_dir=args.experiment_dir,
        checkpoint_name=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
