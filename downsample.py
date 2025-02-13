#!/usr/bin/env python3
from pathlib import Path
import re
import sys
import os
import argparse
from PIL import Image


# Default glob: Match any jpg file that does not end with '_downsampled.jpg'
DEFAULT_REGEX = re.compile(r"^(?!.*_downsampled\.jpg$).*\.jpg$")


def downsample_image(input_path, compression_quality=95):
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to 8-bit grayscale (monochrome)
            img = img.convert("L")
            
            # Resize (downsample) the image to 324x244 pixels using a high-quality resampling filter
            new_size = (324, 244)
            img = img.resize(new_size, resample=Image.LANCZOS)
            
            # Build the output filename by appending '_downsampled' before the extension
            directory, filename = os.path.split(input_path)
            basename, _ = os.path.splitext(filename)
            output_filename = f"{basename}_downsampled.png"
            output_path = os.path.join(directory, output_filename)
            
            # Save the resulting image as a JGP
            img.save(output_path, format="PNG")
            print("Downsampled image saved as:", output_path)
    except Exception as e:
        print("An error occurred:", e)
        sys.exit(1)

def main():

    argparser = argparse.ArgumentParser(description="Downsample an image monochrome to 324x244 pixels")
    argparser.add_argument("input_image_dir", help="The directory containing the input image")
    argparser.add_argument("--regex", 
                           help="The regex pattern to match the input image files", 
                           default=DEFAULT_REGEX)
    args = argparser.parse_args()
    
    input_dir = os.path.abspath(args.input_image_dir)
    regex_pattern = args.regex

    # Verify the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        sys.exit(1)

    print("Input directory:", os.path.abspath(input_dir))
    for filepath in os.listdir(input_dir):
        if re.match(regex_pattern, filepath):
            filepath = os.path.join(input_dir, filepath)
            print("Processing:", filepath)
            downsample_image(filepath)


if __name__ == "__main__":
    main()
