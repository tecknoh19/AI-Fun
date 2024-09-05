# Analyze images and create textual desctipions of the contents.
# This was fun to write, I hope you find it useful

import argparse
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Set up CLI argument parsing
parser = argparse.ArgumentParser(description='Generate captions for images in a directory using BLIP model.')
parser.add_argument('image_dir', type=str, help='Path to the directory containing the images')
parser.add_argument('output_dir', type=str, help='Path to the directory where the captions will be saved')
args = parser.parse_args()

# Load the BLIP model and processor from the local directory
model_dir = "../blip-image-captioning-large"  # Adjust this path to your local model directory
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir)

# Create the output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Directory containing images passed as a CLI argument
image_dir = args.image_dir
output_dir = args.output_dir

# Iterate through the directory and analyze each image
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        # Preprocess and generate text with modified parameters for more detailed descriptions
        inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(
            **inputs,
            min_length=30,
            max_length=150,  # Increase max tokens for more detailed output
            num_beams=25,     # Beam search for better, more optimal captions
            temperature=0.25,  # Lower temperature for more focused and detailed output
            do_sample=True,
            top_p=0.9,        # Nucleus sampling to allow some flexibility
            early_stopping=True,  # Allow for longer descriptions
            repetition_penalty=1.5  # Penalize repetitive text
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Save the generated text to a file with the same name as the image
        text_filename = os.path.splitext(filename)[0] + '.txt'  # Replace .jpg/.png with .txt
        text_file_path = os.path.join(output_dir, text_filename)

        with open(text_file_path, 'w') as text_file:
            text_file.write(generated_text)

        # Output the image description with colored text
        print(f"{Fore.GREEN}Image: {filename} \n{Fore.CYAN}Generated Text: {Fore.YELLOW}{generated_text}\n{Fore.MAGENTA}Saved to: {text_file_path}\n{Style.RESET_ALL}")
