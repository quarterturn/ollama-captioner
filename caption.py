#!/usr/bin/env python3
"""
Ollama image captioner — SFW/NSFW classifier + structured caption generator.

Usage:
    python caption.py [options] --input-dir ./images [--batch-dir batch_0001]

Options:
    --format txt|json       Output format (default: txt)
    --batch-dir NAME        Process only this batch directory (default: all)
    --resume                Resume from checkpoint (default: yes)
    --limit N               Stop after N images (for testing)
"""

import os
import sys
import json
import argparse
import base64
import time
import re
from io import BytesIO

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_API_URL = "http://192.168.0.80:11434/api/generate"
#MODEL = "qwen3.6:27b-q8_0"
MODEL = "qwen3.6:35b-a3b-q8_0"
MAX_TOKENS = 2048
TEMPERATURE = 0.4
TOP_K = 64
TOP_P = 0.95
IMAGE_MAX_DIM = 1024  # longest side, preserve aspect ratio
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_CONFIDENCE_THRESHOLD = 0.80
CHARACTER_CONFIDENCE_THRESHOLD = 0.75
OLLAMA_TIMEOUT = 300  # per-request timeout in seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt():
    """Read prompt from prompt.txt in the script directory."""
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {prompt_path} not found.")
        sys.exit(1)


def resize_image(image):
    """Resize to IMAGE_MAX_DIM on longest side, preserve aspect ratio, no padding."""
    w, h = image.size
    if w <= IMAGE_MAX_DIM and h <= IMAGE_MAX_DIM:
        return image
    scale = IMAGE_MAX_DIM / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def image_to_base64(image):
    """Convert PIL image to base64 string (PNG encoding for lossless)."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_checkpoint(checkpoint_path):
    """Load set of already-processed image keys from checkpoint file."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            print(f"Warning: checkpoint file {checkpoint_path} is corrupt, starting fresh.")
    return set()


def save_checkpoint(checkpoint_path, processed_set):
    """Persist checkpoint to disk."""
    with open(checkpoint_path, "w") as f:
        json.dump(sorted(processed_set), f)


def parse_model_response(raw_text):
    """
    Extract valid JSON from model output. Handles:
    - Bare JSON
    - JSON wrapped in ```json ... ```
    - JSON with surrounding chat text
    """
    # Try direct JSON parse first
    stripped = raw_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences
    # Pattern: ```json\n{...}\n```  or ```\n{...}\n```
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find the first { ... } block in the text
    m = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try to find anything that looks like JSON by finding matching braces
    depth = 0
    start = None
    for i, ch in enumerate(stripped):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(stripped[start:i+1])
                except json.JSONDecodeError:
                    start = None
                break

    return None


def call_ollama(prompt, image_base64, retries=3):
    """Call Ollama generate API, retry on failure."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "stream": False,
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")
            if raw:
                return raw
            else:
                print(f"  [warn] Empty response from Ollama (attempt {attempt + 1})")
        except requests.RequestException as e:
            print(f"  [warn] API error: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ollama image captioner")
    parser.add_argument("input_dir", nargs="?", default="./images",
                        help="Directory containing batch subdirectories")
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                        help="Output format (default: txt)")
    parser.add_argument("--batch-dir", default=None,
                        help="Process only this batch directory")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N images (useful for testing)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.join(os.path.dirname(input_dir), "output")
    checkpoint_path = os.path.join(os.path.dirname(input_dir), "processed.json")

    # Load prompt
    prompt = load_prompt()

    # Discover batch directories
    batches = []
    if args.batch_dir:
        batch_path = os.path.join(input_dir, args.batch_dir)
        if not os.path.isdir(batch_path):
            print(f"Error: batch directory {batch_path} not found.")
            sys.exit(1)
        batches.append(args.batch_dir)
    else:
        for entry in sorted(os.listdir(input_dir)):
            if os.path.isdir(os.path.join(input_dir, entry)):
                batches.append(entry)

    if not batches:
        print("No batch directories found.")
        sys.exit(0)

    # Load checkpoint
    if args.no_resume:
        processed = set()
    else:
        processed = load_checkpoint(checkpoint_path)

    print(f"Model: {MODEL}")
    print(f"Batches: {', '.join(batches)}")
    print(f"Format: {args.format}")
    print(f"Already processed: {len(processed)}")
    print(f"Output root: {output_dir}")
    print()

    # Collect images with their batch subdirectory info
    all_images = []
    for batch in batches:
        batch_path = os.path.join(input_dir, batch)
        for fname in sorted(os.listdir(batch_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_images.append((batch, fname))

    total = len(all_images)
    to_process = [(b, f) for b, f in all_images if f not in processed]
    print(f"Total images: {total}")
    print(f"Images to process: {len(to_process)}")
    if args.limit > 0:
        to_process = to_process[:args.limit]
        print(f"Limited to: {args.limit}")
    print()

    if not to_process:
        print("Nothing to do. All images already processed.")
        sys.exit(0)

    # Process
    sfw_count = 0
    nsfw_count = 0
    error_count = 0
    start_time = time.time()

    for idx, (batch, fname) in enumerate(to_process, 1):
        img_path = os.path.join(input_dir, batch, fname)

        try:
            image = Image.open(img_path)
            image = image.convert("RGB")
        except Exception as e:
            print(f"  [{idx}/{len(to_process)}] SKIP {fname}: cannot open ({e})")
            processed.add(fname)
            error_count += 1
            continue

        resized = resize_image(image)
        img_b64 = image_to_base64(resized)

        raw_response = call_ollama(prompt, img_b64)
        if raw_response is None:
            print(f"  [{idx}/{len(to_process)}] FAIL {fname}: no response from Ollama")
            error_count += 1
            processed.add(fname)
            continue

        parsed = parse_model_response(raw_response)
        if parsed is None:
            print(f"  [{idx}/{len(to_process)}] FAIL {fname}: could not parse JSON")
            print(f"    Raw: {raw_response[:200]}...")
            error_count += 1
            processed.add(fname)
            continue

        # Extract rating and determine routing
        rating = parsed.get("rating", "sfw").lower().strip()
        if rating not in ("sfw", "nsfw"):
            rating = "sfw"  # default

        # Filter by confidence thresholds
        if "captured_text" in parsed:
            parsed["captured_text"] = [
                t for t in parsed["captured_text"]
                if t.get("confidence", 0) >= TEXT_CONFIDENCE_THRESHOLD
            ]

        # Ensure required character fields exist
        if "characters" in parsed:
            for ch in parsed["characters"]:
                ch.setdefault("state_of_dress", "unknown")
                ch.setdefault("visible_body_parts", "unknown")

        # Prepare output
        base_name = os.path.splitext(fname)[0]
        sfw_dir = os.path.join(output_dir, "sfw", batch)
        if rating == "sfw":
            out_dir = sfw_dir
            sfw_count += 1
        else:
            out_dir = os.path.join(output_dir, "nsfw", batch)
            nsfw_count += 1

        os.makedirs(out_dir, exist_ok=True)

        # Write output
        if args.format == "txt":
            caption = parsed.get("subject_and_action", "")
            out_file = os.path.join(out_dir, base_name + ".txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(caption)
        else:
            # JSON format — write the full structured caption
            out_file = os.path.join(out_dir, base_name + ".json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

        # Progress
        elapsed = time.time() - start_time
        per_min = (idx / elapsed * 60) if elapsed > 0 else 0
        eta_total = (elapsed / idx * len(to_process)) if idx > 0 else 0
        eta_str = f" ETA {eta_total/60:.1f}h" if eta_total > 3600 else f" ETA {eta_total/60:.0f}m" if eta_total > 60 else f" ETA {eta_total:.0f}s"
        prefix = "SFW" if rating == "sfw" else "NSF"
        print(f"  [{idx}/{len(to_process)}] {prefix} {fname}  ({per_min:.1f} img/min){eta_str}")

        # Checkpoint after each successful image
        processed.add(fname)
        save_checkpoint(checkpoint_path, processed)

    # Summary
    total_elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print(f"Done!")
    print(f"  Processed: {len(to_process) - error_count}")
    print(f"  SFW:       {sfw_count}")
    print(f"  NSFW:      {nsfw_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Time:      {total_elapsed:.0f}s")
    print(f"  Rate:      {(len(to_process) - error_count) / total_elapsed:.1f} img/s")
    print(f"  Output:    {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
