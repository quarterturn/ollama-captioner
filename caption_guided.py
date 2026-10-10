#!/usr/bin/env python3
\"\"\"
Guided Image Captioner for Mushoku Tensei.
Uses a grounding database and character profiles to provide the vision model with 
specific visual anchors for high-precision identity confirmation.
\"\"\"
import argparse
import base64
import json
import os
import sqlite3
import time
import requests
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_API_URL = \"http://192.168.0.80:11434/api/generate\"
MODEL = \"gemma4-31b-it:latest\"
MAX_TOKENS = 4096
TEMPERATURE = 0.1
OLLAMA_TIMEOUT = 900 

# ---------------------------------------------------------------------------
# JSON Validation & Cleaning (Borrowed from openrouter-image-captioner)
# ---------------------------------------------------------------------------
VALID_RATINGS = {\"sfw\", \"nsfw\"}
REQUIRED_TOP_FIELDS = [\"rating\", \"medium\", \"subject_and_action\", \"style_description\", \"characters\"]

def try_parse_caption_json(text):
    \"\"\"Parse model output JSON. Handles extra trailing text after closing brace.\"\"\"
    if not text:
        return None
    clean = re.sub(r'[\u200b\ufeff]', '', text.strip())
    for marker in [\"```json\", \"```\"]:
        if clean.startswith(marker):
            clean = clean[len(marker):]
            if clean.endswith(\"```\"):
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
        if ch == '\\\\':
            escape_next = True
            continue
        if ch == '\"':
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
    \"\"\"Return (is_valid, reason_string) for the parsed model JSON.\"\"\"
    if not isinstance(cap_json, dict):
        return False, \"root is not a dict\"
    for field in REQUIRED_TOP_FIELDS:
        if cap_json.get(field) is None:
            return False, f\"missing required field '{field}'\"
    rt = (cap_json.get(\"rating\") or \"\").lower()
    if rt not in VALID_RATINGS:
        return False, f\"invalid rating '{rt}'\"
    if not isinstance(cap_json.get(\"medium\"), str) or not cap_json[\"medium\"].strip():
        return False, \"medium missing or empty\"
    sd = cap_json.get(\"style_description\")
    if not isinstance(sd, dict):
        return False, \"style_description is not a dict\"
    for sub in [\"color_palette\", \"mood\", \"lighting\"]:
        if sd.get(sub) is None and sub != \"color_palette\":
            return False, f\"style_description.{sub} missing\"
    chs = cap_json.get(\"characters\")
    if not isinstance(chs, list) or len(chs) == 0:
        return False, \"characters empty or not a list\"
    return True, None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_db_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_profiles(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.endswith(',\\n}'): content = content.replace(',\\n}', '}')
        elif content.endswith(',}'): content = content.replace(',}', '}')
        return json.loads(content)

def image_to_base64(path):
    with open(path, \"rb\") as f:
        return base64.b64encode(f.read()).decode(\"utf-8\")

def call_ollama(prompt, image_b64):
    payload = {
        \"model\": MODEL,
        \"prompt\": prompt,
        \"images\": [image_b64],
        \"max_tokens\": MAX_TOKENS,
        \"temperature\": TEMPERATURE,
        \"stream\": False,
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json().get(\"response\", \"\")
    except Exception as e:
        print(f\"API Error: {e}\")
        return None

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=\"Guided Anime Captioner\")
    parser.add_argument(\"--db\", required=True, help=\"Path to the identity SQL database\")
    parser.add_argument(\"--profiles\", required=True, help=\"Path to char_profiles.json\")
    parser.add_argument(\"--prompt\", required=True, help=\"Path to prompt.txt\")
    parser.add_argument(\"--input\", required=True, help=\"Input images directory\")
    parser.add_argument(\"--output\", default=None, help=\"Optional: Directory to save captions. If omitted, saves beside images.\")
    parser.add_argument(\"--limit\", type=int, default=0, help=\"Limit number of images\")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    with open(args.prompt, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    db_conn = get_db_conn(args.db)
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    try:
        rows = db_conn.execute(\"SELECT name, path, characters FROM images\").fetchall()
    except sqlite3.OperationalError as e:
        print(f\"DB Error: {e}\")
        return

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f\"Processing {len(rows)} images with {MODEL}...\")

    for row in tqdm(rows):
        img_name = row['name']
        img_path = row['path']
        char_list = row['characters']

        injection = \"\"
        if char_list:
            names = [n.strip() for n in char_list.split(',') if n.strip()]
            for name in names:
                desc = profiles.get(name, \"No specific description available.\")
                injection += f\"- **{name}**: {desc}\\n\"
        else:
            injection = \"No specific characters identified for this image.\"

        final_prompt = prompt_template.replace(
            \"The following characters are guaranteed to be in this image. Here are their names and distinguishing features:\",
            f\"The following characters are guaranteed to be in this image. Here are their names and distinguishing features:\\n\\n{injection}\"
        )

        if not os.path.exists(img_path):
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
                        # Determine output path: use provided output dir or the image's own directory
                        if args.output:
                            out_dir = args.output
                        else:
                            out_dir = os.path.dirname(img_path)
                        
                        out_file = os.path.join(out_dir, f\"{os.path.splitext(img_name)[0]}.json\")
                        with open(out_file, 'w', encoding='utf-8') as f:
                            json.dump(cap_json, f, indent=2, ensure_ascii=False)
                        success = True
                    else:
                        print(f\"Validation fail for {img_name} (Attempt {attempts}): {reason}\")
                else:
                    print(f\"JSON parse fail for {img_name} (Attempt {attempts})\")
        
        if not success:
            print(f\"FAILED to get valid caption for {img_name} after {max_attempts} attempts.\")

    db_conn.close()
    print(\"Done!\")

if __name__ == \"__main__\":
    main()
