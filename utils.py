import os
import numpy as np
import h5py
import json
import torch
import random
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import torchvision.transforms as transforms
from PIL import Image

# Constants
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

def create_word_map(captions, min_word_freq=5):
    """
    Create a word map from captions.
    
    :param captions: list of captions
    :param min_word_freq: minimum frequency of words to be included
    :return: word map
    """
    word_freq = Counter()
    for caption in tqdm(captions):
        word_freq.update(caption.split())
    
    # Create word map
    word_map = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNK_TOKEN: 3}
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    for i, w in enumerate(words):
        word_map[w] = i + 4  # +4 because of the special tokens
    
    return word_map

def process_caption(caption, word_map, max_length=100):
    """
    Process a caption: lowercase, remove punctuation, etc.
    
    :param caption: caption string
    :param word_map: word map
    :param max_length: maximum caption length
    :return: processed caption
    """
    # Simple preprocessing
    caption = caption.lower().strip()
    
    # Tokenize
    tokens = caption.split()
    
    # Encode caption with word map
    encoded = []
    encoded.append(word_map[START_TOKEN])
    
    for word in tokens:
        if word in word_map:
            encoded.append(word_map[word])
        else:
            encoded.append(word_map[UNK_TOKEN])
    
    encoded.append(word_map[END_TOKEN])
    
    # Pad to max_length
    if len(encoded) < max_length:
        encoded.extend([word_map[PAD_TOKEN]] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length - 1] + [word_map[END_TOKEN]]  # Truncate and add END token
    
    return encoded

def get_transform(train=True, image_size=256):
    """
    Get image transform for training or evaluation.
    
    :param train: training or evaluation mode
    :param image_size: image size
    :return: image transform
    """
    # Normalization parameters for pretrained ResNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def decode_caption(encoded_caption, word_map_reverse):
    """
    Decode a caption from indices to words.
    
    :param encoded_caption: encoded caption
    :param word_map_reverse: reversed word map (indices to words)
    :return: decoded caption
    """
    decoded = []
    for ind in encoded_caption:
        word = word_map_reverse[ind]
        if word == START_TOKEN:
            continue
        elif word == END_TOKEN:
            break
        decoded.append(word)
    
    return ' '.join(decoded)

def create_input_files(train_image_dir, train_captions_file, test_image_dir, 
                      test_file, output_folder, min_word_freq=5,
                      max_len=100, captions_per_image=5):
    """
    Create input files for training decoder.
    
    :param train_image_dir: folder with training images
    :param train_captions_file: CSV file with training captions
    :param test_image_dir: folder with test images
    :param test_file: CSV file with test image IDs
    :param output_folder: folder to save files
    :param min_word_freq: words occurring less frequently than this threshold are binned as <unk>
    :param max_len: maximum length of captions
    :param captions_per_image: number of captions to sample per image
    :return: None
    """
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read training captions
    try:
        with open(train_captions_file, 'r', encoding='utf-8') as f:
            import csv
            reader = csv.reader(f)
            header = next(reader)
            # Extract image_id and caption columns
            image_id_index = header.index('image_id')
            caption_index = header.index('caption')
            
            train_image_captions = {}
            for row in reader:
                if len(row) <= max(image_id_index, caption_index):
                    continue  # Skip rows with insufficient columns
                image_id = str(row[image_id_index])
                caption = str(row[caption_index])
                
                if image_id not in train_image_captions:
                    train_image_captions[image_id] = []
                train_image_captions[image_id].append(caption)
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(train_captions_file, 'r', encoding='latin-1') as f:
            import csv
            reader = csv.reader(f)
            header = next(reader)
            # Extract image_id and caption columns
            image_id_index = header.index('image_id')
            caption_index = header.index('caption')
            
            train_image_captions = {}
            for row in reader:
                if len(row) <= max(image_id_index, caption_index):
                    continue  # Skip rows with insufficient columns
                image_id = str(row[image_id_index])
                caption = str(row[caption_index])
                
                if image_id not in train_image_captions:
                    train_image_captions[image_id] = []
                train_image_captions[image_id].append(caption)
    
    # Read test image IDs
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            import csv
            reader = csv.reader(f)
            header = next(reader)
            image_id_index = header.index('image_id')
            
            test_image_ids = [str(row[image_id_index]) for row in reader if len(row) > image_id_index]
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(test_file, 'r', encoding='latin-1') as f:
            import csv
            reader = csv.reader(f)
            header = next(reader)
            image_id_index = header.index('image_id')
            
            test_image_ids = [str(row[image_id_index]) for row in reader if len(row) > image_id_index]
    
    print(f"Found {len(train_image_captions)} training images with captions")
    print(f"Found {len(test_image_ids)} test images")
    
    # Create word map
    all_captions = []
    for captions in train_image_captions.values():
        all_captions.extend(captions)
    
    word_map = create_word_map(all_captions, min_word_freq=min_word_freq)
    
    # Create reverse word map
    word_map_reverse = {v: k for k, v in word_map.items()}
    
    # Save word map
    with open(os.path.join(output_folder, 'word_map.json'), 'w', encoding='utf-8') as f:
        json.dump(word_map, f)
    
    # Process training data
    train_image_paths = []
    train_encoded_captions = []
    train_caption_lengths = []
    
    for image_id, captions in tqdm(train_image_captions.items(), desc='Processing train data'):
        image_path = os.path.join(train_image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue
        
        # Sample captions_per_image captions per image
        if len(captions) >= captions_per_image:
            sampled_captions = sample(captions, captions_per_image)
        else:
            sampled_captions = captions + [choice(captions) for _ in range(captions_per_image - len(captions))]
        
        # Encode sampled captions
        for caption in sampled_captions:
            encoded_caption = process_caption(caption, word_map, max_len)
            train_image_paths.append(image_path)
            train_encoded_captions.append(encoded_caption)
            train_caption_lengths.append(len([t for t in encoded_caption if t != word_map[PAD_TOKEN]]))
    
    # Save training data
    with open(os.path.join(output_folder, 'train_image_paths.json'), 'w', encoding='utf-8') as f:
        json.dump(train_image_paths, f)
    
    with open(os.path.join(output_folder, 'train_encoded_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(train_encoded_captions, f)
    
    with open(os.path.join(output_folder, 'train_caption_lengths.json'), 'w', encoding='utf-8') as f:
        json.dump(train_caption_lengths, f)
    
    # Process test data
    test_image_paths = []
    
    for image_id in tqdm(test_image_ids, desc='Processing test data'):
        image_path = os.path.join(test_image_dir, f"{image_id}.jpg")
        if os.path.exists(image_path):
            test_image_paths.append(image_path)
    
    # Save test data
    with open(os.path.join(output_folder, 'test_image_paths.json'), 'w', encoding='utf-8') as f:
        json.dump(test_image_paths, f)
    
    print("Created input files at %s" % output_folder)
    print(f"Number of training examples: {len(train_image_paths)}")
    print(f"Number of test examples: {len(test_image_paths)}")
    print(f"Size of vocabulary: {len(word_map)}")

def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to avoid explosion of gradients.
    
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip) 