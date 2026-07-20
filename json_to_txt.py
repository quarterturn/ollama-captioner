#!/usr/bin/env python3
"""
Convert structured JSON captions back to plaintext for image models that don't use JSON.

Usage:
    python json_to_txt.py <input-dir> [--batch-dir BATCH] [--format narrative|comma-separated]

Input dir should contain sfw/ and nsfw/ subdirectories with batch_X folders of .json files.
Outputs mirror the input structure but with .txt files instead.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

# Default settings
DEFAULT_FORMAT = "narrative"
DEFAULT_SEPARATOR = "; "


def load_json(path: str) -> Any:
    """Load a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Cannot read {path}: {e}")
        return None


def narrative_format(data: Dict[str, Any], separator: str = DEFAULT_SEPARATOR) -> str:
    """Convert a JSON caption to a flowing narrative paragraph."""
    parts: List[str] = []

    # Medium
    med = data.get("medium", "")
    if med:
        parts.append(f"{med.capitalize()}")

    # Mood and lighting
    sd = data.get("style_description", {})
    mood = sd.get("mood", "")
    lighting = sd.get("lighting", "")
    time_of_day = sd.get("time_of_day", "")
    if mood or time_of_day or lighting:
        scene_bits = []
        if time_of_day:
            scene_bits.append(time_of_day.replace("_", " "))
        if lighting:
            scene_bits.append(lighting.replace("_", " "))
        if mood:
            scene_bits.append(mood.replace("_", " ") + " mood")
        parts.append(f". A {' '.join(scene_bits)} scene.")

    # Subject and action (main paragraph)
    subject = data.get("subject_and_action", "")
    if subject:
        parts.append(f"{separator}{subject}".replace(". ", ".  "))

    # Characters with state_of_dress and visible_body_parts woven in
    chars = data.get("characters", [])
    for i, ch in enumerate(chars):
        char_parts = []
        pos_marker = ch.get("description", "")  # already includes position
        state = ch.get("state_of_dress", "")
        visible = ch.get("visible_body_parts", "")
        action = ch.get("action", "")
        gaze = ch.get("gaze", "")
        lighting_on_subject = ch.get("lighting_on_subject", "")

        if state:
            state = state.replace("_", " ").capitalize()
            if not state.endswith("."):
                state += "."

        # Build character description
        char_descs = []
        if pos_marker:
            char_descs.append(pos_marker)
        if state:
            char_descs.append(state)
        if visible:
            char_descs.append(f"Visible: {visible.replace('_', ' ')}")
        if action:
            char_descs.append(f"{action}")
        if gaze:
            char_descs.append(f"Gaze: {gaze}")
        if lighting_on_subject:
            char_descs.append(f"Lighting: {lighting_on_subject}")

        if char_descs:
            char_text = f"{DEFAULT_SEPARATOR}{DEFAULT_SEPARATOR.join(char_descs)}"
            parts.append(char_text)

    # Captured text as dialogue
    text_entries = data.get("captured_text", [])
    for entry in text_entries:
        raw = entry.get("text", "")
        lang = entry.get("language", "")
        conf = entry.get("confidence", 0)
        if raw and conf >= 0.80:
            parts.append(f"{DEFAULT_SEPARATOR}Captured text (confidence {conf:.2f}, lang {lang}): '{raw}'")

    # Camera details
    cam = sd.get("camera", {})
    if cam:
        cam_parts = []
        if cam.get("angle"):
            cam_parts.append(f"Camera angle: {cam['angle'].replace('_', ' ')}")
        if cam.get("framing"):
            cam_parts.append(f"Framing: {cam['framing'].replace('_', ' ')}")
        if cam.get("depth_of_field"):
            cam_parts.append(f"Depth of field: {cam['depth_of_field'].replace('_', ' ')}")
        if cam.get("focus_area"):
            cam_parts.append(f"Focus area: {cam['focus_area'].replace('_', ' ')}")
        if cam_parts:
            parts.append(f"{DEFAULT_SEPARATOR}Camera: {', '.join(cam_parts)}")

    # Post-processing
    pp = data.get("post_processing", {})
    if pp:
        pp_parts = []
        if pp.get("effects"):
            pp_parts.append(f"Effects: {pp['effects']}")
        if pp.get("filters"):
            pp_parts.append(f"Filters: {pp['filters']}")
        if pp_parts:
            parts.append(f"{DEFAULT_SEPARATOR}Post-processing: {', '.join(pp_parts)}")

    # Join all parts into one cohesive paragraph
    result = " ".join(parts)
    if result.startswith(". "):
        result = result[2:]
    return result.lstrip()


def comma_separated_format(data: Dict[str, Any]) -> str:
    """Convert to a comma-separated phrase list (useful for some models like Qwen image gen)."""
    phrases: List[str] = []

    # Medium and style
    med = data.get("medium", "")
    if med:
        phrases.append(med.replace("_", " ").capitalize())

    sd = data.get("style_description", {})
    if sd.get("mood"):
        phrases.append(sd["mood"].replace("_", " "))
    if sd.get("lighting"):
        phrases.append(sd["lighting"].replace("_", " "))
    if sd.get("time_of_day"):
        phrases.append(sd["time_of_day"].replace("_", " "))

    # Subject as main phrase
    subject = data.get("subject_and_action", "")
    if subject:
        sentences = [s.strip() for s in subject.split(".") if s.strip()]
        for sent in sentences:
            if not sent.endswith("."):
                sentences[sentences.index(sent)] = sent + "."

    # Characters with state of dress and visible body parts
    chars = data.get("characters", [])
    for i, ch in enumerate(chars):
        char_phrases = []
        if ch.get("description"):
            char_phrases.append(f"Character {i+1}: {ch['description']}")
        if ch.get("state_of_dress"):
            char_phrases.append(f"State of dress: {ch['state_of_dress']}")
        if ch.get("visible_body_parts"):
            char_phrases.append(f"Visible body parts: {ch['visible_body_parts'].replace('_', ' ')}")
        if ch.get("action"):
            char_phrases.append(f"Action: {ch['action']}")
        if ch.get("gaze"):
            char_phrases.append(f"Gaze: {ch['gaze'].replace('_', ' ')}")
        if char_phrases:
            for p in char_phrases:
                phrases.append(p.replace(",", " |"))  # Avoid comma confusion

    # Captured text
    text_entries = data.get("captured_text", [])
    for entry in text_entries:
        raw = entry.get("text", "")
        lang = entry.get("language", "")
        conf = entry.get("confidence", 0)
        if raw and conf >= 0.80:
            phrases.append(f"Captured text ({lang} conf={conf:.2f}): '{raw}'")

    # Camera work
    cam = sd.get("camera", {})
    if cam:
        if cam.get("angle"):
            phrases.append(f"Camera angle: {cam['angle'].replace('_', ' ')}")
        if cam.get("framing"):
            phrases.append(f"Framing: {cam['framing'].replace('_', ' ')}")
        if cam.get("depth_of_field"):
            phrases.append(f"Depth of field: {cam['depth_of_field'].replace('_', ' ')}")
        if cam.get("focus_area"):
            phrases.append(f"Focus area: {cam['focus_area'].replace('_', ' ')}")

    # Post-processing
    pp = data.get("post_processing", {})
    if pp:
        if pp.get("effects"):
            phrases.append(f"Effects: {pp['effects'].replace('_', ' ')}")
        if pp.get("filters"):
            phrases.append(f"Filters: {pp['filters'].replace('_', ' ')}")

    # Compositing into a coherent paragraph
    if not phrases:
        return ""

    # First phrase is typically the medium, rest are supporting details
    result = f"{phrases[0]}"
    for i, phrase in enumerate(phrases[1:], 1):
        if i % 3 == 0:  # Every 3rd phrase gets a period instead of comma for readability
            result += f"  {phrase}"
        else:
            result += f", {phrase}"
    return result


def convert_json_to_txt(json_input_dir: str) -> None:
    """Convert all JSON captions in an input directory structure to .txt files."""
    input_path = Path(json_input_dir)
    output_dir = input_path / "_txt_output"
    output_dir.mkdir(exist_ok=True)

    # Find all JSON files recursively
    json_files = sorted(input_path.rglob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {json_input_dir}")
        return

    for json_file in json_files:
        data = load_json(str(json_file))
        if not data:
            continue

        # Determine output path (mirror input structure but under txt_output)
        rel_path = json_file.relative_to(json_input_dir)
        txt_rel_path = rel_path.with_suffix(".txt")
        txt_path = output_dir / txt_rel_path
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert based on format
        if FORMAT == "narrative":
            txt_content = narrative_format(data)
        elif FORMAT == "comma-separated":
            txt_content = comma_separated_format(data)
        else:
            print(f"ERROR: Unknown format: {FORMAT}")
            continue

        # Write the TXT file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert structured JSON captions to plaintext for image models.",
        epilog="Example: python json_to_txt.py /path/to/json_output --format narrative --output-dir /path/to/txt_output"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing JSON caption files"
    )
    parser.add_argument(
        "--format",
        choices=["narrative", "comma-separated"],
        default="narrative",
        help="Output format (default: narrative)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for TXT files (default: input_dir/_txt_output)",
    )
    ARGS = parser.parse_args()
    FORMAT = ARGS.format
    OUTPUT_DIR = ARGS.output_dir if ARGS.output_dir else f"{ARGS.input_dir}_txt_output"
    convert_json_to_txt(ARGS.input_dir)
    print(f"Converted {len(list(Path(ARGS.input_dir).rglob('*.json')))} JSON files to TXT format '{FORMAT}'")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
