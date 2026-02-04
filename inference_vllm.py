"""
vLLM Inference for Forest Fire Detection VLM
Supports both merged models and LoRA adapters
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Union, List
import argparse

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class ForestFireVLM:
    """Forest Fire Detection VLM using vLLM for fast inference"""
    
    def __init__(
        self,
        model_path: str,
        use_lora: bool = False,
        base_model: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the VLM for inference
        
        Args:
            model_path: Path to exported model (merged or LoRA adapter)
            use_lora: If True, load as LoRA adapter (requires base_model)
            base_model: Base model name (required if use_lora=True)
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        self.model_path = Path(model_path)
        self.use_lora = use_lora
        
        # Load metadata if available
        metadata_path = self.model_path / "export_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"Loaded model metadata: {self.metadata}")
        else:
            self.metadata = {}
        
        # Determine which model to load
        if use_lora:
            if base_model is None:
                # Try to get from metadata
                base_model = self.metadata.get('base_model')
                if base_model is None:
                    raise ValueError("base_model must be provided when use_lora=True")
            
            print(f"Loading base model: {base_model}")
            print(f"LoRA adapter: {model_path}")
            model_to_load = base_model
            self.lora_path = str(model_path)
        else:
            print(f"Loading merged model: {model_path}")
            model_to_load = str(model_path)
            self.lora_path = None
        
        # Initialize vLLM
        print("Initializing vLLM...")
        self.llm = LLM(
            model=model_to_load,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192,  # Match training max_length
            enable_lora=use_lora,
        )
        
        print("✓ Model loaded successfully!")
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict:
        """
        Analyze an image for forest fire detection
        
        Args:
            image_path: Path to image file
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with fire detection results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create prompt (same as training)
        prompt = """Analyze the aerial image from the UAV's camera and detect potential forest fires. Answer the following questions based on the image analysis
in a JSON schema. If a question cannot be determined from the image answer with 'Cannot be determined'.

  forest_fire_smoke_visible: ['Yes', 'No']
  forest_fire_flames_visible: ['Yes', 'No']
  confirm_uncontrolled_forest_fire: ['Yes', 'Closer investigation required', 'No forest fire visible']
  fire_state: ['Ignition Phase', 'Growth Phase', 'Fully Developed Phase', 'Decay Phase', 'Cannot be determined', 'No forest fire visible']
  fire_type: ['Ground Fire', 'Surface Fire', 'Crown Fire', 'Cannot be determined', 'No forest fire visible']
  fire_intensity: ['Low', 'Moderate', 'High', 'Cannot be determined', 'No forest fire visible']
  fire_size: ['Small', 'Medium', 'Large', 'Cannot be determined', 'No forest fire visible']
  fire_hotspots: ['Multiple hotspots', 'One hotspot', 'Cannot be determined', 'No forest fire visible']
  infrastructure_nearby: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
  people_nearby: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
  tree_vitality: ['Vital', 'Moderate Vitality', 'Declining', 'Dead', 'Cannot be determined', 'No forest fire visible']
"""
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>"],
        )
        
        # Create input with image
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        
        # Generate with LoRA if applicable
        if self.use_lora:
            lora_request = LoRARequest("forest_fire_lora", 1, self.lora_path)
            outputs = self.llm.generate(
                inputs,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        else:
            outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        
        # Parse response
        response_text = outputs[0].outputs[0].text.strip()
        
        try:
            result = json.loads(response_text)
            result['_raw_response'] = response_text
            result['_image_path'] = str(image_path)
            return result
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            return {
                "_error": "JSON parse error",
                "_raw_response": response_text,
                "_image_path": str(image_path),
            }
    
    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[dict]:
        """
        Analyze multiple images in batch (faster than sequential)
        
        Args:
            image_paths: List of image paths
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of detection results
        """
        results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path, temperature, max_tokens)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Forest Fire VLM Inference with vLLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to exported model (merged or LoRA adapter)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image to analyze"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory of images to analyze in batch"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.jsonl",
        help="Output file for predictions (JSONL format)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Load as LoRA adapter (requires --base-model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (required if --use-lora)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image_dir must be provided")
    
    # Initialize model
    print("=" * 80)
    print("FOREST FIRE VLM INFERENCE")
    print("=" * 80)
    
    vlm = ForestFireVLM(
        model_path=args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Collect images to process
    if args.image:
        image_paths = [args.image]
    else:
        image_dir = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([
            str(p) for p in image_dir.rglob("*")
            if p.suffix.lower() in exts
        ])
        print(f"\nFound {len(image_paths)} images in {args.image_dir}")
    
    # Process images
    print(f"\nProcessing {len(image_paths)} image(s)...")
    results = vlm.analyze_batch(
        image_paths=image_paths,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    fire_detected = sum(
        1 for r in results
        if r.get('confirm_uncontrolled_forest_fire') == 'Yes'
    )
    investigation_needed = sum(
        1 for r in results
        if r.get('confirm_uncontrolled_forest_fire') == 'Closer investigation required'
    )
    no_fire = sum(
        1 for r in results
        if r.get('confirm_uncontrolled_forest_fire') == 'No forest fire visible'
    )
    
    print(f"Total images: {len(results)}")
    print(f"Fire detected: {fire_detected}")
    print(f"Investigation needed: {investigation_needed}")
    print(f"No fire: {no_fire}")
    
    # Show first result as example
    if results:
        print("\nExample result:")
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
