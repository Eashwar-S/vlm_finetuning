import os, json, base64, time, hashlib, asyncio
from pathlib import Path
import requests
import aiohttp
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "google/gemini-3-pro-preview"

PROMPT_TEXT = """Analyze the aerial image from the UAV's camera and detect potential forest fires. Answer the following questions based on the image analysis
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

def img_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # crude mime guess
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"

def validate_answer(ans: dict) -> tuple[bool, str]:
    # Minimal validator: required keys + enum membership.
    # (You can swap in jsonschema library later.)
    if not isinstance(ans, dict):
        return False, "Answer is not a dict"
    for k in RESPONSE_SCHEMA["required"]:
        if k not in ans:
            return False, f"Missing key: {k}"
        allowed = RESPONSE_SCHEMA["properties"][k]["enum"]
        if ans[k] not in allowed:
            return False, f"Bad value for {k}: {ans[k]} (allowed: {allowed})"
    extra = set(ans.keys()) - set(RESPONSE_SCHEMA["properties"].keys())
    if extra:
        return False, f"Extra keys: {sorted(extra)}"
    return True, "ok"

def call_openrouter(image_path: str, prompt_text: str, json_schema: dict) -> dict:
    """Synchronous API call (kept for backwards compatibility)"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "crossfire-dataset-builder",
    }
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_to_data_url(image_path)}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fire_schema", "strict": True, "schema": json_schema},
        },
        "temperature": 0.0,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content) if isinstance(content, str) else content


async def call_openrouter_async(session: aiohttp.ClientSession, image_path: str, prompt_text: str, json_schema: dict) -> dict:
    """Async API call for concurrent processing"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "crossfire-dataset-builder",
    }
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_to_data_url(image_path)}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fire_schema", "strict": True, "schema": json_schema},
        },
        "temperature": 0.0,
    }
    
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
        response.raise_for_status()
        data = await response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else content

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

async def process_image_async(session: aiohttp.ClientSession, image_path: Path, image_dir: Path, done: set) -> tuple[bool, dict, str]:
    """Process a single image asynchronously with retries"""
    sha = file_sha1(str(image_path))
    
    if sha in done:
        return False, None, None  # Already processed
    
    last_err = None
    for attempt in range(3):
        try:
            ans = await call_openrouter_async(session, str(image_path), PROMPT_TEXT, RESPONSE_SCHEMA)
            ok, msg = validate_answer(ans)
            
            if not ok:
                last_err = msg
                await asyncio.sleep(1.0)
                continue
            
            row = {
                "image": str(image_path.relative_to(image_dir)),
                "filename": image_path.name,
                "sha1": sha,
                "prompt": PROMPT_TEXT,
                "response_schema": RESPONSE_SCHEMA,
                "teacher_answer": ans,
                "teacher_model": MODEL,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            
            return True, row, None  # Success
            
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(2.0)
    
    return False, None, last_err  # Failed after retries


async def build_jsonl_async(image_dir: str, out_jsonl: str, max_concurrent: int = 5):
    """
    Build JSONL dataset with concurrent API calls
    
    Args:
        image_dir: Directory containing images
        out_jsonl: Output JSONL file path
        max_concurrent: Maximum number of concurrent API calls (default: 5)
    """
    image_dir = Path(image_dir)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip already processed files if re-running
    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sha1"])
                except Exception:
                    pass
    
    print(f"Already processed: {len(done)} images")

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])
    
    # Filter out already processed
    pending_paths = [p for p in paths if file_sha1(str(p)) not in done]
    print(f"Total images found: {len(paths)}")
    print(f"Remaining to process: {len(pending_paths)}")
    print(f"Concurrent API calls: {max_concurrent}")
    print("-" * 80)
    
    if not pending_paths:
        print("✓ All images already processed!")
        return
    
    # Process images concurrently
    success_count = 0
    fail_count = 0
    
    async with aiohttp.ClientSession() as session:
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(path):
            async with semaphore:
                return await process_image_async(session, path, image_dir, done)
        
        # Open output file
        with out_path.open("a", encoding="utf-8") as f_out:
            # Process in batches to avoid overwhelming the API
            tasks = [process_with_semaphore(p) for p in pending_paths]
            
            for coro in asyncio.as_completed(tasks):
                success, row, error = await coro
                
                if success:
                    f_out.write(json.dumps(row) + "\n")
                    f_out.flush()
                    done.add(row["sha1"])
                    success_count += 1
                    print(f"✓ OK ({success_count}/{len(pending_paths)}): {row['filename']}")
                elif error:
                    fail_count += 1
                    print(f"✗ FAIL ({fail_count}): {error}")
    
    print("-" * 80)
    print(f"\n✓ Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total processed: {len(done)}")


def build_jsonl(image_dir: str, out_jsonl: str, max_concurrent: int = 1):
    """
    Build JSONL dataset (wrapper for backward compatibility)
    
    Args:
        image_dir: Directory containing images
        out_jsonl: Output JSONL file path
        max_concurrent: Maximum concurrent API calls (1 = sequential, 5+ = parallel)
    """
    if max_concurrent > 1:
        # Use async version for concurrent processing
        asyncio.run(build_jsonl_async(image_dir, out_jsonl, max_concurrent))
    else:
        # Use synchronous version for sequential processing
        image_dir = Path(image_dir)
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip already processed files if re-running
        done = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["sha1"])
                    except Exception:
                        pass
        print(f"Done: {len(done)}")

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])

        with out_path.open("a", encoding="utf-8") as f_out:
            for p in paths:
                sha = file_sha1(str(p))
                if sha in done:
                    continue

                # retries for schema issues / transient errors
                last_err = None
                for attempt in range(3):
                    try:
                        ans = call_openrouter(str(p), PROMPT_TEXT, RESPONSE_SCHEMA)
                        ok, msg = validate_answer(ans)
                        if not ok:
                            last_err = msg
                            time.sleep(1.0)
                            continue

                        row = {
                            "image": str(p.relative_to(image_dir)),
                            "filename": p.name,
                            "sha1": sha,
                            "prompt": PROMPT_TEXT,
                            "response_schema": RESPONSE_SCHEMA,
                            "teacher_answer": ans,
                            "teacher_model": MODEL,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        f_out.write(json.dumps(row) + "\n")
                        f_out.flush()
                        done.add(sha)
                        print("OK:", p.name)
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(2.0)

                else:
                    print("FAIL:", p.name, "|", last_err)


if __name__ == "__main__":
    # Example usage:
    # Sequential (original behavior):
    # build_jsonl("dataset/images/", "dataset/data/train.jsonl", max_concurrent=1)
    
    # Concurrent (5 parallel API calls - MUCH FASTER!):
    build_jsonl("dataset/images/", "dataset/data/train.jsonl", max_concurrent=3)
    
    # For testing with small dataset:
    # build_jsonl("dataset/images_short/", "dataset/data/train_short.jsonl", max_concurrent=3)
