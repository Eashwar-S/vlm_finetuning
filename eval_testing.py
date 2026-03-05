import os
import json
import tinker
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SAMPLER_PATH = "tinker://dd9265a5-bca1-57fe-9dde-9fc05fb079d9:train:0/sampler_weights/006500"
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def run_inference(image_bytes: bytes, prompt_text: str, sampling_client, tokenizer) -> dict:
    """Run VLM inference on image bytes and return parsed JSON"""
    
    # Build multimodal prompt (image + text)
    model_input = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(
                tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")
            ),
            tinker.types.ImageChunk(data=image_bytes, format="jpeg"),
            tinker.types.EncodedTextChunk(
                tokens=tokenizer.encode(
                    f"<|vision_end|>{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
                )
            ),
        ]
    )

    # Run sampling
    result_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=512,
            temperature=0.0,   # deterministic JSON
        ),
    )

    # Wait for the future to complete and get the result
    result = result_future.result()

    # Decode model output
    output_text = tokenizer.decode(result.sequences[0].tokens)

    # Parse JSON
    try:
        # Remove markdown code fences if present
        cleaned_text = output_text
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0]
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1].split("```")[0]
        
        # Remove special tokens like <|im_end|>
        cleaned_text = cleaned_text.replace("<|im_end|>", "").strip()
        
        # Find JSON object
        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}") + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = cleaned_text[json_start:json_end]
            json_data = json.loads(json_str)
            return json_data
        else:
            return {"error": "Could not find JSON object in output", "raw": output_text}
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}", "raw": output_text}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print("Evaluating VLM Model on ForestFireInsights-Eval")
    print("=" * 80)
    print(f"Checkpoint: {SAMPLER_PATH}")
    
    print("\n📦 Loading dataset (leon-se/ForestFireInsights-Eval)...")
    ds = load_dataset("leon-se/ForestFireInsights-Eval", split="train")
    
    print("\n📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=SAMPLER_PATH,
        base_model=BASE_MODEL,
    )
    print("✅ Model loaded successfully\n")
    
    results = []
    
    for i in tqdm(range(len(ds)), desc="Processing samples"):
        sample = ds[i]
        
        # Extract features
        image = sample['image']
        prompt_text = sample['prompt']
        filename = sample['filename']
        gt_answer = sample['gt_answer']
        
        if isinstance(gt_answer, str):
            try:
                # Some JSON strings might contain single quotes incorrectly or need basic cleaning
                gt_answer = gt_answer.replace("'", '"')
                gt_answer = json.loads(gt_answer)
            except Exception as e:
                print(f"Error parsing gt_answer for {filename}: {e}")
                gt_answer = {}
                
        # Convert image to JPEG bytes
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()
        
        # Run inference
        pred_dict = run_inference(image_bytes, prompt_text, sampling_client, tokenizer)
        # print(f'pred_dict: {pred_dict}')
        # print(f'gt_answer: {gt_answer}')
        # Build result row
        row = {
            "filename": filename,
            "prompt": prompt_text,
            "raw_prediction": json.dumps(pred_dict) if pred_dict else "",
            "raw_gt": json.dumps(gt_answer) if gt_answer else "",
        }
        
        correct_count = 0
        total_questions = 0
        
        if isinstance(gt_answer, dict) and gt_answer:
            for k, v_gt in gt_answer.items():
                v_pred = pred_dict.get(k, None) if isinstance(pred_dict, dict) else None
                
                # Case-insensitive comparison
                is_correct = str(v_gt).strip().lower() == str(v_pred).strip().lower()
                
                row[f"gt_{k}"] = v_gt
                row[f"pred_{k}"] = v_pred
                row[f"correct_{k}"] = is_correct
                
                if is_correct:
                    correct_count += 1
                total_questions += 1
                
        row["sample_accuracy"] = correct_count / total_questions if total_questions > 0 else 0
        results.append(row)
        
    print("\n💾 Saving results...")
    df = pd.DataFrame(results)
    
    # Optional: re-order columns so correct_* are grouped together or easily readable
    # Basic output will just save everything
    output_path = "evaluation_results.xlsx"
    df.to_excel(output_path, index=False)
    
    print(f"✅ Saved detailed results to {output_path}")
    
    print("\n📊 Evaluation Summary:")
    print("=" * 40)
    
    correct_cols = [c for c in df.columns if c.startswith("correct_")]
    for col in correct_cols:
        question = col.replace("correct_", "")
        acc = df[col].astype(bool).mean() * 100
        print(f"  {question:<35} : {acc:5.1f}%")
        
    overall_mean = df["sample_accuracy"].mean() * 100
    print("-" * 40)
    print(f"  {'Overall Average Accuracy':<35} : {overall_mean:5.1f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()