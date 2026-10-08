# Ollama Image Captioner

This project provides tools for generating high-precision structured captions for anime datasets. It leverages vision-language models to create rich, natural-language descriptions for training modern image generation models.

## Guided Captioning Pipeline (`caption_guided.py`)

The guided captioner is designed for high-precision identity grounding. Instead of relying on a model's general knowledge, it injects specific visual anchors from a grounding database into the prompt.

### How it Works
1. **Grounding:** It reads a SQLite database (e.g., from the `ccip_grounding` project) to identify which characters are present in an image.
2. **Profile Injection:** It maps those characters to a `char_profiles.json` file containing specific visual traits (e.g., "mole under left eye").
3. **Prompt Synthesis:** These traits are injected directly into the system prompt, forcing the model to verify specific physical markers before naming a character.
4. **Structured Output:** The model outputs a single valid JSON object containing rating, mood, lighting, and detailed per-character descriptions.

### Usage
```bash
python3 caption_guided.py \
  --db "/path/to/mushoku_tensei_3.sql" \
  --profiles "/path/to/char_profiles.json" \
  --prompt "/path/to/prompt.txt" \
  --input "/path/to/images" \
  --output "/path/to/captions" \
  --limit 10
```

### Requirements
- `pillow`: Image processing
- `requests`: API communication with Ollama
- `tqdm`: Progress tracking

## Files
- `caption_guided.py`: The main guided captioning script.
- `char_profiles.json`: Visual trait database for target characters.
- `prompt.txt`: The system instruction set for the vision model.
- `requirements.txt`: Python dependencies.
