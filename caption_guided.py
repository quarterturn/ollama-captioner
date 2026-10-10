#!/usr/bin/env python3
"""
Guided Image Captioner for Mushoku Tensei.
Uses a grounding database and character profiles to provide the vision model with 
specific visual anchors for high-precision identity confirmation.
"""
import argparse
import base64
import json
import os
import sqlite3
import time
import requests
import re
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_API_URL = "http://192.168.0.80:11434/api/generate"
MODEL = "gemma4-31b-it:latest"
MAX_TOKENS = 4096
TEMPERATURE = 0.1
OLLAMA_TIMEOUT = 900 

# ---------------------------------------------------------------------------
# JSON Validation & Cleaning
# ---------------------------------------------------------------------------
VALID_RATINGS = {"sfw", "nsfw"}
REQUIRED_TOP_FIELDS = ["rating", "medium", "subject_and_action", "style_description", "characters"]

def try_parse_caption_json(text):
    """Parse model output JSON. Handles extra trailing text after closing brace."""
    if not text:
        return None
    clean = re.sub(r'[\u200b\ufeff]', '', text.strip())
    for marker in ["```json", "```"]:
        if clean.startswith(marker):
            clean = clean[len(marker):]
            if clean.endswith("```"):
                clean = clean[:-3].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    start = clean.find('{')
    if start < 0:
        return None
    brace_depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(clean)):
        ch = clean[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    try:
                        return json.loads(clean[start:i+1])
                    except json.JSONDecodeError:
                        return None
    return None

def validate_caption_json(cap_json):
    """Return (is_valid, reason_string) for the parsed model JSON."""
    if not isinstance(cap_json, dict):
        return False, "root is not a dict"
    for field in REQUIRED_TOP_FIELDS:
        if cap_json.get(field) is None:
            return False, f"missing required field '{field}'"
    rt = (cap_json.get("rating") or "").lower()
    if rt not in VALID_RATINGS:
        return False, f"invalid rating '{rt}'"
    if not isinstance(cap_json.get("medium"), str) or not cap_json["medium"].strip():
        return False, "medium missing or empty"
    sd = cap_json.get("style_description")
    if not isinstance(sd, dict):
        return False, "style_description is not a dict"
    for sub in ["color_palette", "mood", "lighting"]:
        if sd.get(sub) is None and sub != "color_palette":
            return False, f"style_description.{sub} missing"
    chs = cap_json.get("characters")
    if not isinstance(chs, list) or len(chs) == 0:
        return False, "characters empty or not a list"
    return True, None

# ---------------------------------------------------------------------------
# Database & Helpers
# ---------------------------------------------------------------------------
def get_db_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def db_init(conn):
    """Initialize progress tracking tables and columns."""
    try:
        conn.execute("ALTER TABLE images ADD COLUMN is_captioned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        # Column probably already exists
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS captions (
            filename TEXT PRIMARY KEY,
            caption TEXT NOT NULL DEFAULT '',
            rating TEXT DEFAULT '',
            char_name TEXT DEFAULT '',
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            model TEXT DEFAULT '',
            status TEXT DEFAULT 'complete',
            error_msg TEXT DEFAULT '',
            created_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.commit()

def save_result(conn, img_name, output, result, char_list):
    """Save caption to DB and write .json file beside the original image."""
    if isinstance(result, dict) and "error" in result:
        status = "failed"
        error_msg = result["error"][:200]
        caption_text = ""
        pt = ct = 0
        model_used = MODEL
        rating_val = ""
    else:
        cap_json = try_parse_caption_json(output)
        if cap_json is None:
            status = "failed"
            error_msg = "Model returned non-parseable JSON"
            caption_text = (output or "")[:200]
            pt = ct = 0
            model_used = MODEL
            rating_val = ""
        else:
            status = "complete"
            error_msg = ""
            caption_text = json.dumps(cap_json, ensure_ascii=False, indent=2)
            pt = result.get("prompt_tokens", 0) if isinstance(result, dict) else 0
            ct = result.get("completion_tokens", 0) if isinstance(result, dict) else 0
            model_used = MODEL
            rating_val = cap_json.get("rating", "")

    conn.execute("""
        INSERT OR REPLACE INTO captions 
        (filename, caption, rating, char_name, prompt_tokens, completion_tokens, model, status, error_msg, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
    """, (img_name, caption_text, rating_val, char_list or "", pt, ct, model_used, status, error_msg))

    if status == "complete":
        conn.execute("UPDATE images SET is_captioned=1 WHERE name=?", (img_name,))
    
    conn.commit()
    return status == "complete"

def load_profiles(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.endswith(',\n}'): content = content.replace(',\n}', '}')
        elif content.endswith(',}'): content = content.replace(',}', '}')
        return json.loads(content)

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_ollama(prompt, image_b64):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"API Error: {e}")
        return None

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Guided Anime Captioner")
    parser.add_argument("--db", required=True, help="Path to the identity SQL database")
    parser.add_argument("--profiles", required=True, help="Path to char_profiles.json")
    parser.add_argument("--prompt-a", required=True, help="Path to the first part of the prompt (prompt-a.txt)")
    parser.add_argument("--prompt-b", required=True, help="Path to the second part of the prompt (prompt-b.txt)")
    parser.add_argument("--input", required=True, help="Input images directory")
    parser.add_argument("--output", default=None, help="Optional: Directory to save captions and images. If omitted, saves beside original images.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    with open(args.prompt_a, 'r', encoding='utf-8') as f:
        prompt_a = f.read()
    with open(args.prompt_b, 'r', encoding='utf-8') as f:
        prompt_b = f.read()

    db_conn = get_db_conn(args.db)
    db_init(db_conn)
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    try:
        # Only fetch images that are not yet captioned
        rows = db_conn.execute("SELECT name, path, characters FROM images WHERE is_captioned = 0 AND (characters IS NOT NULL AND characters != '')").fetchall()
    except sqlite3.OperationalError as e:
        print(f"DB Error: {e}")
        return

    total_to_process = len(rows)
    if args.limit > 0:
        rows = rows[:args.limit]
        total_to_process = len(rows)

    print(f"Found {len(rows)} uncaptioned images to process (out of {total_to_process if args.limit == 0 else 'limited set'})...")

    newly_captioned = 0
    failed = 0

    pbar = tqdm(rows, desc="Captioning")
    for row in pbar:
        img_name = row['name']
        img_path = row['path']
        char_list = row['characters']

        injection = ""
        if char_list:
            names = [n.strip() for n in char_list.split(',') if n.strip()]
            for name in names:
                desc = profiles.get(name, "No specific description available.")
                injection += f"- **{name}**: {desc}\\n"
        else:
            injection = "No specific characters identified for this image."

        final_prompt = f"{prompt_a}\\n\\n{injection}\\n\\n{prompt_b}"

        if not os.path.exists(img_path):
            pbar.set_postfix({"status": f"Missing: {img_name}"})
            continue
            
        success = False
        attempts = 0
        max_attempts = 2
        
        img_b64 = image_to_base64(img_path)
        
        while attempts < max_attempts and not success:
            attempts += 1
            raw_output = call_ollama(final_prompt, img_b64)
            
            if raw_output:
                cap_json = try_parse_caption_json(raw_output)
                if cap_json:
                    valid, reason = validate_caption_json(cap_json)
                    if valid:
                        if args.output:
                            out_dir = args.output
                        else:
                            out_dir = os.path.dirname(img_path)
                        
                        out_file = os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}.json")
                        with open(out_file, 'w', encoding='utf-8') as f:
                            json.dump(cap_json, f, indent=2, ensure_ascii=False)
                        
                        img_dest = os.path.join(out_dir, img_name)
                        if img_path != img_dest:
                            shutil.copy2(img_path, img_dest)
                        
                        # Save progress to DB
                        if save_result(db_conn, img_name, raw_output, {"prompt_tokens": 0, "completion_tokens": 0}, char_list):
                            success = True
                            newly_captioned += 1
                    else:
                        print(f"Validation fail for {img_name} (Attempt {attempts}): {reason}")
                else:
                    print(f"JSON parse fail for {img_name} (Attempt {attempts})")
        
        if not success:
            print(f"FAILED to get valid caption for {img_name} after {max_attempts} attempts.")
            # Log the failure in the DB as well
            save_result(db_conn, img_name, None, {"error": "Max retries reached or validation failed"}, char_list)
            failed += 1
            
        pbar.set_postfix({"OK": newly_captioned, "Fail": failed})

    db_conn.close()
    print(f"\\nDone! Newly captioned: {newly_captioned}, Failed: {failed}")

if __name__ == "__main__":
    main()
