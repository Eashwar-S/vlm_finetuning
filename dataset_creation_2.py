import os, json, base64, time, hashlib, asyncio, random
from pathlib import Path
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


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def img_to_data_url_bytes(image_bytes: bytes, suffix: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"

def validate_answer(ans: dict) -> tuple[bool, str]:
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

async def call_openrouter_async(
    session: aiohttp.ClientSession,
    image_data_url: str,
    prompt_text: str,
    json_schema: dict,
) -> dict:
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
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fire_schema", "strict": True, "schema": json_schema},
        },
        "temperature": 0.0,
    }

    async with session.post(url, headers=headers, json=payload) as resp:
        # Handle rate limits explicitly
        if resp.status == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RuntimeError(f"RATE_LIMIT: retry_after={retry_after}")
        resp.raise_for_status()
        data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else content

async def process_one(
    session: aiohttp.ClientSession,
    image_path: Path,
    image_dir: Path,
    sha: str,
    image_data_url: str,
    max_attempts: int = 5,
) -> tuple[bool, dict | None, str | None]:
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            ans = await call_openrouter_async(session, image_data_url, PROMPT_TEXT, RESPONSE_SCHEMA)
            ok, msg = validate_answer(ans)
            if not ok:
                last_err = msg
                # small backoff for schema mismatch
                await asyncio.sleep(0.6 + random.random() * 0.4)
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
            return True, row, None

        except Exception as e:
            s = str(e)
            last_err = s

            # Rate limit: respect Retry-After if present
            if "RATE_LIMIT" in s:
                # parse retry_after=...
                ra = None
                try:
                    ra = s.split("retry_after=")[1].strip()
                    ra = float(ra) if ra not in ("None", "") else None
                except Exception:
                    ra = None
                sleep_s = ra if ra is not None else min(10.0, 1.5 * (2 ** (attempt - 1)))
            else:
                sleep_s = min(10.0, 1.0 * (2 ** (attempt - 1)))

            # jitter
            sleep_s = sleep_s * (0.7 + random.random() * 0.6)
            await asyncio.sleep(sleep_s)

    return False, None, last_err

async def build_jsonl_async(image_dir: str, out_jsonl: str, max_concurrent: int = 10):
    image_dir = Path(image_dir)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load done SHAs once
    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sha1"])
                except Exception:
                    pass

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])

    # Precompute sha once; skip done without re-hashing multiple times
    pending = []
    for p in all_paths:
        sha = file_sha1(str(p))
        if sha not in done:
            pending.append((p, sha))

    print(f"Already processed: {len(done)}")
    print(f"Total images found: {len(all_paths)}")
    print(f"Remaining to process: {len(pending)}")
    print(f"Workers (concurrency): {max_concurrent}")
    print("-" * 80)

    if not pending:
        print("✓ All images already processed!")
        return

    # A bounded queue keeps memory stable and avoids creating thousands of tasks
    q: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue(maxsize=max_concurrent * 3)
    results: asyncio.Queue[tuple[bool, dict | None, str | None]] = asyncio.Queue()

    timeout = aiohttp.ClientTimeout(total=180, connect=30, sock_read=180)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2, ttl_dns_cache=300, enable_cleanup_closed=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def producer():
            for p, sha in pending:
                await q.put((p, sha))
            # sentinels to stop workers
            for _ in range(max_concurrent):
                await q.put(None)

        async def worker(worker_id: int):
            while True:
                item = await q.get()
                if item is None:
                    q.task_done()
                    return
                p, sha = item
                try:
                    # Read + build data URL ONCE per image (reused across retries)
                    img_bytes = p.read_bytes()
                    data_url = img_to_data_url_bytes(img_bytes, p.suffix)

                    ok, row, err = await process_one(session, p, image_dir, sha, data_url)
                    await results.put((ok, row, err))
                finally:
                    q.task_done()

        async def writer():
            success = 0
            fail = 0
            total = len(pending)

            with out_path.open("a", encoding="utf-8") as f_out:
                while success + fail < total:
                    ok, row, err = await results.get()
                    if ok and row:
                        f_out.write(json.dumps(row) + "\n")
                        f_out.flush()
                        done.add(row["sha1"])
                        success += 1
                        print(f"✓ OK ({success}/{total}): {row['filename']}")
                    else:
                        fail += 1
                        print(f"✗ FAIL ({fail}/{total}): {err}")

            print("-" * 80)
            print("✓ Processing complete!")
            print(f"  Success: {success}")
            print(f"  Failed:  {fail}")
            print(f"  Total processed (incl prior): {len(done)}")

        prod_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker(i)) for i in range(max_concurrent)]
        writer_task = asyncio.create_task(writer())

        await prod_task
        await q.join()
        await writer_task
        for t in worker_tasks:
            await t

def build_jsonl(image_dir: str, out_jsonl: str, max_concurrent: int = 1):
    if max_concurrent > 1:
        asyncio.run(build_jsonl_async(image_dir, out_jsonl, max_concurrent=max_concurrent))
    else:
        raise NotImplementedError("Use your existing sequential version if you still need it.")

if __name__ == "__main__":
    build_jsonl("dataset/images/", "dataset/data/train.jsonl", max_concurrent=5)