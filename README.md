---
license: cc-by-nc-4.0
---
Ollama image captioner

A simple python script which uses an Ollama API endpoint to engage a local language model which has vision capabilities, such as Gemma3 27B.
I recommend Gemma3 27B because it follows instructions well, is able to read text within an image, and fits on consumer hardware like a 3090 with a reasonable quant.

Install:
1. clone the repo
2. create a conda or python env for the project
3. activate the env
4. install the dependencies via 'pip install -r requirements.txt'
5. edit prompt.txt to your liking
6. if ollama is not running on the same machine as where the script is ran, edit caption.py 'OLLAMA_API_URL = "http://localhost:11434/api/generate"' to whatever IP or hostname Ollama is running on
7. edit caption.py 'model": "gemma3:27b-it-q8_0",' to match the vision-capable model you intend to use
note:
'ollama ls' will show you the available models:
```
$ ollama ls
NAME                                    ID              SIZE     MODIFIED
gemma3:27b-it-q8_0                      273cbcd67032    29 GB    2 days ago
```
8. put the images to be captioned into 'images' in .png or .jpg format

Creating a prompt:
The example prompt.txt shows a format which works reliably for me, in terms of creating a caption suitable for a LoRA training dataset for natural language image and video models. You will need to adjust the prompt to your specific task, and you should definitely proofread the results of the script. I suggest using something like https://github.com/hassan-sd/manual-image-captioner to easily compare the caption to the image. Gemma3 27B does a good job, but it still sometimes parrots back the character guide descriptions or hallucinates.