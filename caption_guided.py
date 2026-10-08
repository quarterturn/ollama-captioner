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
# Helpers
# ---------------------------------------------------------------------------
def get_db_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_profiles(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        # Handle potential trailing commas for flexibility
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
    parser.add_argument("--prompt", required=True, help="Path to prompt.txt")
    parser.add_argument("--input", required=True, help="Input images directory")
    parser.add_argument("--output", required=True, help="Output directory for JSON captions")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    args = parser.parse_args()

    # Load Assets
    profiles = load_profiles(args.profiles)
    with open(args.prompt, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    db_conn = get_db_conn(args.db)
    os.makedirs(args.output, exist_ok=True)

    # Get images from DB that aren't captioned yet (assuming a simple images table)
    # We look for the images table created by ccip_grounder.py
    try:
        rows = db_conn.execute("SELECT name, path, characters FROM images").fetchall()
    except sqlite3.OperationalError as e:
        print(f"DB Error: {e}")
        return

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Processing {len(rows)} images with {MODEL}...")

    for row in tqdm(rows):
        img_name = row['name']
        img_path = row['path']
        char_list = row['characters'] # Comma separated string

        # 1. Build Character Injection Block
        injection = ""
        if char_list:
            names = [n.strip() for n in char_list.split(',') if n.strip()]
            for name in names:
                desc = profiles.get(name, "No specific description available.")
                injection += f"- **{name}**: {desc}\\n"
        else:
            injection = "No specific characters identified for this image."

        # 2. Assemble Final Prompt
        # Look for the specific insertion point in the prompt
        final_prompt = prompt_template.replace(
            "The following characters are guaranteed to be in this image. Here are their names and distinguishing features:",
            f"The following characters are guaranteed to be in this image. Here are their names and distinguishing features:\\n\\n{injection}"
        )

        # 3. Execute Vision Task
        if not os.path.exists(img_path):
            continue
            
        img_b64 = image_to_base64(img_path)
        raw_output = call_ollama(final_prompt, img_b64)

        if raw_output:
            # Clean and Save
            # Remove markdown fences if the model ignored the prompt
            clean_json = raw_output.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:].strip()
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3].strip()
            
            try:
                # Validate JSON before saving
                parsed = json.loads(clean_json)
                out_file = os.path.join(args.output, f"{os.path.splitext(img_name)[0]}.json")
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                print(f"JSON Error for {img_name}: {raw_output[:100]}...")

    db_conn.close()
    print("Done!")

if __name__ == "__main__":
    main()
