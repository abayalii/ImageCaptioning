import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from models import Encoder, DecoderWithAttention
from utils import get_transform, decode_caption

def caption_images_from_checkpoint(checkpoint_path, word_map_path, test_folder, output_csv, beam_size=3):
    """
    Caption images and create a submission file using a checkpoint file.
    
    :param checkpoint_path: path to the checkpoint file
    :param word_map_path: path to word map JSON file
    :param test_folder: folder with test images
    :param output_csv: path to output CSV file
    :param beam_size: beam size for beam search
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load word map
    with open(word_map_path, 'r', encoding='utf-8') as f:
        word_map = json.load(f)
    
    # Create reverse word map
    word_map_reverse = {v: k for k, v in word_map.items()}
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    
    # Move models to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Get transform
    transform = get_transform(train=False)
    
    # Get all image files in test folder
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Results list
    results = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Captioning images"):
        # Get image ID (filename without extension)
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(test_folder, image_file)
        
        # Process image
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        # Encode image
        with torch.no_grad():
            encoded_image = encoder(img)
        
        # Generate caption with beam search
        with torch.no_grad():
            seq, _ = decoder.caption_image(encoded_image, word_map, beam_size=beam_size)
        
        # Convert sequence to words
        caption = decode_caption(seq, word_map_reverse)
        
        # Add to results
        results.append({'image_id': image_id, 'caption': caption})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Try to convert image_id to integer if possible
    try:
        df['image_id'] = df['image_id'].astype(int)
    except ValueError:
        print("Warning: Could not convert image_id to integer. Keeping as string.")
    
    df.to_csv(output_csv, index=False)
    print(f"Created submission file: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Caption images and create submission file using checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--word_map', type=str, required=True, help='Path to word map JSON file')
    parser.add_argument('--test_folder', type=str, default='test/test', help='Folder with test images')
    parser.add_argument('--output', type=str, default='submission.csv', help='Path to output CSV file')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search')
    args = parser.parse_args()
    
    caption_images_from_checkpoint(
        checkpoint_path=args.checkpoint, 
        word_map_path=args.word_map,
        test_folder=args.test_folder,
        output_csv=args.output,
        beam_size=args.beam_size
    )

if __name__ == '__main__':
    main() 