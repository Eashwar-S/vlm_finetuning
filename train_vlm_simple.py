"""
Simple Forest Fire Detection VLM Training using Tinker Cookbook
Works with the installed version of tinker_cookbook
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Any
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import chz
from tinker_cookbook.recipes.sl_basic import chat_datasets, train
from tinker_cookbook.renderers import TrainOnWhat


@chz.chz
class ForestFireDatasetBuilder:
    """Builder for forest fire detection dataset"""
    
    jsonl_path: str = "dataset/data/train.jsonl"
    images_dir: str = "dataset/images"
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    renderer_name: str = "qwen3"
    
    batch_size: int = 1
    max_length: int = 8192
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    
    def load_conversations(self) -> list[dict]:
        """Load and convert JSONL to conversation format"""
        conversations = []
        images_dir = Path(self.images_dir)
        
        print(f"Loading data from {self.jsonl_path}...")
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Get image
                    image_filename = data.get('image') or data.get('filename')
                    if not image_filename:
                        continue
                    
                    image_path = images_dir / image_filename
                    if not image_path.exists():
                        print(f"Warning: Image not found: {image_path}")
                        continue
                    
                    # Load image
                    pil_image = Image.open(image_path).convert('RGB')
                    
                    # Create conversation in tinker format
                    # For VLMs, we need to pass the image through the conversation
                    conversation = {
                        "messages": [
                            {
                                "role": "user",
                                "content": data['prompt'],
                                "image": pil_image  # Pass PIL image directly
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(data['teacher_answer'])
                            }
                        ]
                    }
                    
                    conversations.append(conversation)
                    
                    if (idx + 1) % 10 == 0:
                        print(f"Loaded {idx + 1} examples...")
                        
                except Exception as e:
                    print(f"Error loading example {idx}: {e}")
                    continue
        
        print(f"Successfully loaded {len(conversations)} conversations")
        return conversations
    
    def __call__(self):
        """Build the dataset"""
        from tinker_cookbook.supervised.types import SupervisedDataset
        from tinker_cookbook.renderers import get_renderer, Message
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        import random
        import math
        
        conversations = self.load_conversations()
        
        # Get renderer
        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer(name=self.renderer_name, tokenizer=tokenizer)
        
        # Create a custom dataset class
        class ForestFireDataset(SupervisedDataset):
            def __init__(self, conversations, renderer, config):
                self.conversations = conversations
                self.renderer = renderer
                self.config = config
                self.indices = list(range(len(conversations)))
                random.shuffle(self.indices)
            
            def get_batch(self, batch_idx: int) -> list:
                """Get a batch of training data"""
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(self.indices))
                
                batch = []
                for i in range(start_idx, end_idx):
                    conv_idx = self.indices[i]
                    conversation = self.conversations[conv_idx]
                    
                    # Convert messages to Message format
                    messages = []
                    for msg in conversation["messages"]:
                        # For VLM, the image should be handled by the renderer
                        # We'll pass it as part of the content
                        message = Message(
                            role=msg["role"],
                            content=msg["content"]
                        )
                        # Store image separately if present
                        if "image" in msg:
                            message["image"] = msg["image"]
                        messages.append(message)
                    
                    # Convert conversation to datum
                    datum = chat_datasets.conversation_to_datum(
                        conversation=messages,
                        renderer=self.renderer,
                        max_length=self.config.max_length,
                        train_on_what=self.config.train_on_what,
                    )
                    batch.append(datum)
                
                return batch
            
            def set_epoch(self, seed: int = 0):
                """Shuffle data for new epoch"""
                random.seed(seed)
                random.shuffle(self.indices)
            
            def __len__(self):
                """Number of batches"""
                return math.ceil(len(self.conversations) / self.config.batch_size)
        
        dataset = ForestFireDataset(conversations, renderer, self)
        
        # Return (train_dataset, eval_dataset)
        return dataset, None


def main():
    """Main training function"""
    
    print("=" * 80)
    print("Starting Forest Fire Detection VLM Training")
    print("=" * 80)
    print("\nℹ️  Checkpoints will be saved to: ./experiments/forest_fire_vlm/")
    print("ℹ️  After training, export checkpoints with: python export_model.py")
    print("ℹ️  Run inference with: python inference_vllm.py")
    print("=" * 80)
    
    # Create dataset builder
    dataset_builder = ForestFireDatasetBuilder(
        jsonl_path="dataset/data/train.jsonl",
        images_dir="dataset/images",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3",
        batch_size=1,
        max_length=8192,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    
    # Build training configuration
    config = train.Config(
        log_path="./experiments/forest_fire_vlm_5k",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        dataset_builder=dataset_builder,
        learning_rate=1e-5,
        lr_schedule="constant",  # Options: "linear" or "constant"
        num_epochs=3,
        lora_rank=16,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=500,
        eval_every=500,
        wandb_project="forest-fire-vlm",  # Set to your project name to enable W&B
        wandb_name="forest-fire-vlm-training",
    )
    
    print(f"Model: {config.model_name}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"LR Schedule: {config.lr_schedule}")
    print("=" * 80)
    
    # Run training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
