---
license: cc-by-nc-4.0
---
Ollama image captioner v 2.0

A simple python script which uses an Ollama API endpoint to engage a local language model which has vision capabilities, such as Qwen 3.6 35B A3B. It produces natural language captions in either plaintext or json format, and will classify images as SFW or NSFW. The prompt is designed to prevent the model from being vague about captioning NSFW material. The script tracks processed file in processed.json so you can cancel and resume where you left off, if needed.

Install:
1. clone the repo
2. create a conda or python env for the project
3. activate the env
4. install the dependencies via 'pip install -r requirements.txt'
5. edit prompt.txt to your liking
6. if ollama is not running on the same machine as where the script is ran, edit caption.py 'OLLAMA_API_URL = "http://localhost:11434/api/generate"' to whatever IP or hostname Ollama is running on
7. edit caption.py 'model": "qwen3.6:35b-a3b-q8_0",' to match the vision-capable model you intend to use
8. put the images to be captioned into 'images' in .png or .jpg format

```
Usage:
    python caption.py [options] --input-dir ./images [--batch-dir batch_0001]

Options:
    --format txt|json       Output format (default: txt)
    --batch-dir NAME        Process only this batch directory (default: all)
    --resume                Resume from checkpoint (default: yes)
    --limit N               Stop after N images (for testing)
```
