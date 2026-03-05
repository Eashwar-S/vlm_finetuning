import os
import json
import base64
import argparse
from io import BytesIO
from datasets import load_dataset
import pandas as pd
from tqdm.asyncio import tqdm
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

RESPONSE_SCHEMA = {
  "type": "object",
  "properties": {
    "forest_fire_smoke_visible": {"type": "string", "enum": ["Yes", "No"]},
    "forest_fire_flames_visible": {"type": "string", "enum": ["Yes", "No"]},
    "confirm_uncontrolled_forest_fire": {"type": "string", "enum": ["Yes", "Closer investigation required", "No forest fire visible"]},
    "fire_state": {"type": "string", "enum": ["Ignition Phase", "Growth Phase", "Fully Developed Phase", "Decay Phase", "Cannot be determined", "No forest fire visible"]},
    "fire_type": {"type": "string", "enum": ["Ground Fire", "Surface Fire", "Crown Fire", "Cannot be determined", "No forest fire visible"]},
    "fire_intensity": {"type": "string", "enum": ["Low", "Moderate", "High", "Cannot be determined", "No forest fire visible"]},
    "fire_size": {"type": "string", "enum": ["Small", "Medium", "Large", "Cannot be determined", "No forest fire visible"]},
    "fire_hotspots": {"type": "string", "enum": ["Multiple hotspots", "One hotspot", "Cannot be determined", "No forest fire visible"]},
    "infrastructure_nearby": {"type": "string", "enum": ["Yes", "No", "Cannot be determined", "No forest fire visible"]},
    "people_nearby": {"type": "string", "enum": ["Yes", "No", "Cannot be determined", "No forest fire visible"]},
    "tree_vitality": {"type": "string", "enum": ["Vital", "Moderate Vitality", "Declining", "Dead", "Cannot be determined", "No forest fire visible"]},
  },
  "required": [
    "forest_fire_smoke_visible","forest_fire_flames_visible","confirm_uncontrolled_forest_fire",
    "fire_state","fire_type","fire_intensity","fire_size","fire_hotspots",
    "infrastructure_nearby","people_nearby","tree_vitality"
  ],
  "additionalProperties": False
}


def piltobase64(image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


async def call_openrouter_async(session: aiohttp.ClientSession, image_b64: str, prompt_text: str, model: str) -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "crossfire-gemini-eval",
    }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fire_schema", "strict": True, "schema": RESPONSE_SCHEMA},
        },
        "temperature": 0.0,
    }
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
        response.raise_for_status()
        data = await response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else content


async def process_sample_async(session: aiohttp.ClientSession, sample, model_id: str) -> dict:
    image = sample['image']
    prompt_text = sample['prompt']
    filename = sample['filename']
    gt_answer = sample['gt_answer']
    
    if isinstance(gt_answer, str):
        try:
            gt_answer = gt_answer.replace("'", '"')
            gt_answer = json.loads(gt_answer)
        except Exception as e:
            print(f"Error parsing gt_answer for {filename}: {e}")
            gt_answer = {}
            
    # Convert image to Base64
    image_b64 = piltobase64(image)
    
    # Run inference with retries
    pred_dict = None
    last_err = None
    for attempt in range(3):
        try:
            pred_dict = await call_openrouter_async(session, image_b64, prompt_text, model_id)
            break
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(2.0)
            
    row = {
        "filename": filename,
        "prompt": prompt_text,
        "raw_prediction": json.dumps(pred_dict) if pred_dict else "",
        "raw_gt": json.dumps(gt_answer) if gt_answer else "",
        "error": last_err if pred_dict is None else ""
    }
    
    correct_count = 0
    total_questions = 0
    
    if isinstance(gt_answer, dict) and gt_answer:
        for k, v_gt in gt_answer.items():
            v_pred = pred_dict.get(k, None) if isinstance(pred_dict, dict) else None
            is_correct = str(v_gt).strip().lower() == str(v_pred).strip().lower()
            
            row[f"gt_{k}"] = v_gt
            row[f"pred_{k}"] = v_pred
            row[f"correct_{k}"] = is_correct
            
            if is_correct:
                correct_count += 1
            total_questions += 1
            
    row["sample_accuracy"] = correct_count / total_questions if total_questions > 0 else 0
    return row


async def evaluate_model_async(model_id: str, output_prefix: str, ds, max_concurrent: int = 15):
    print("=" * 80)
    print(f"Evaluating {model_id} via OpenRouter concurrently")
    print("=" * 80)
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(sample):
            async with semaphore:
                return await process_sample_async(session, sample, model_id)
        
        tasks = [process_with_semaphore(ds[i]) for i in range(len(ds))]
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {model_id}"):
            row = await coro
            results.append(row)
            
    print("\n💾 Saving results...")
    df = pd.DataFrame(results)
    
    output_path = f"{output_prefix}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Saved detailed results to {output_path}")
    
    # Save the summary statistics
    correct_cols = [c for c in df.columns if c.startswith("correct_")]
    summary_data = []
    
    for col in correct_cols:
        question = col.replace("correct_", "")
        if df[col].notna().any():
            acc = df[col].astype(bool).mean() * 100
        else:
            acc = 0.0
            
        summary_data.append({"Category": question, f"{output_prefix}_Model": f"{acc:.1f}%"})
        
    overall_mean = df["sample_accuracy"].mean() * 100
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.loc[len(summary_df)] = ["Overall Average Accuracy", f"{overall_mean:.1f}%"]
    
    # Save to Excel
    output_summary_path = f"{output_prefix}_summary.xlsx"
    summary_df.to_excel(output_summary_path, index=False)
    print(f"✅ Saved summary to {output_summary_path}\n")

    print("📊 Evaluation Summary:")
    print("=" * 40)
    print(summary_df.to_markdown(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini via OpenRouter.")
    parser.add_argument("--model", type=str, choices=["flash", "pro", "both"], default="both", help="Model to evaluate")
    parser.add_argument("--concurrent", type=int, default=15, help="Number of concurrent API requests")
    args = parser.parse_args()
    
    print("\n📦 Loading dataset (leon-se/ForestFireInsights-Eval)...")
    ds = load_dataset("leon-se/ForestFireInsights-Eval", split="train")

    if args.model in ["flash", "both"]:
        asyncio.run(evaluate_model_async("google/gemini-3-flash-preview", "evaluation_results_gemini_flash", ds, args.concurrent))

    if args.model in ["pro", "both"]:
        asyncio.run(evaluate_model_async("google/gemini-3-pro-preview", "evaluation_results_gemini_pro", ds, args.concurrent))


if __name__ == "__main__":
    main()
