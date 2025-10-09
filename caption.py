import os
import requests
from PIL import Image
import base64
from io import BytesIO
import json

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to resize image to 896x896, scaling based on horizontal axis
def resize_image(image):
    original_width, original_height = image.size
    target_size = 896
    
    # Scale based on width to maintain aspect ratio
    scale_factor = target_size / original_width
    new_height = int(original_height * scale_factor)
    
    # Resize image
    resized_image = image.resize((target_size, new_height), Image.Resampling.LANCZOS)
    
    # If height is not 896, pad to make it 896
    if new_height != target_size:
        new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        offset = (0, (target_size - new_height) // 2)
        new_image.paste(resized_image, offset)
        return new_image
    return resized_image

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Directory containing the images
image_directory = "./images"

# Read prompt from prompt.txt
try:
    with open("prompt.txt", "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read().strip()
except FileNotFoundError:
    print("Error: prompt.txt file not found in the script directory.")
    prompt = ""  # Fallback to empty prompt or could exit
    # Optionally, you could exit the script here: 
    # sys.exit(1)
except Exception as e:
    print(f"Error reading prompt.txt: {e}")
    prompt = ""
    # Optionally, you could exit the script here: 
    # sys.exit(1)

# Iterate through the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_directory, filename)
        print(f"Processing image: {filename}")
        
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            continue
        
        # Resize image to 896x896
        resized_image = resize_image(image)
        
        # Convert image to base64 for Ollama API
        image_base64 = image_to_base64(resized_image)
        
        # Check if prompt is empty before proceeding
        if not prompt:
            print(f"Skipping {filename}: No valid prompt available.")
            continue
        
        # Prepare payload for Ollama API
        payload = {
            "model": "gemma3:27b-it-q8_0",
            "prompt": prompt,
            "images": [image_base64],
            "max_tokens": 512,
            "temperature": 0.6,
            "top_k": 64,
            "top_p": 0.95,
            "stop": ["<end_of_turn>"],
            "stream": False  # Explicitly disable streaming for single response
        }
        
        # Send request to Ollama API
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            
            # Log the full response for debugging
            response_text = response.text
            print(f"Raw API response for {filename}: {response_text}")
            
            # Try parsing as single JSON first
            try:
                response_json = response.json()
                generated_text = response_json.get("response", "")
            except ValueError:
                # Handle streaming response (multiple JSON objects)
                generated_text = ""
                for line in response_text.splitlines():
                    try:
                        json_obj = json.loads(line)
                        if "response" in json_obj:
                            generated_text += json_obj["response"]
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line for {filename}: {line}")
                        continue
            
            if not generated_text:
                print(f"Warning: Empty or missing 'response' for {filename}")
                continue
                
            # Print and save the generated text
            print(f"Caption for: {filename}")
            print(generated_text)
            print("*---------------------------------------------------*")
            
            # Save to file
            output_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(image_directory, output_filename), "w") as file:
                file.write(generated_text)
                
        except requests.RequestException as e:
            print(f"API error for {filename}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error for {filename}: {e}")
            continue