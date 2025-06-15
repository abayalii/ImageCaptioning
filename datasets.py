import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from utils import get_transform

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train' or 'test'
        :param transform: image transform pipeline
        """
        self.split = split
        assert split in {'train', 'test'}

        # Open files
        with open(os.path.join(data_folder, f'{split}_image_paths.json'), 'r', encoding='utf-8') as f:
            self.image_paths = json.load(f)

        if split == 'train':
            # Load captions and caption lengths for training
            with open(os.path.join(data_folder, 'train_encoded_captions.json'), 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
                
            with open(os.path.join(data_folder, 'train_caption_lengths.json'), 'r', encoding='utf-8') as f:
                self.caption_lengths = json.load(f)

        # Load word map
        with open(os.path.join(data_folder, 'word_map.json'), 'r', encoding='utf-8') as f:
            self.word_map = json.load(f)
            
        # Create reverse word map
        self.word_map_reverse = {v: k for k, v in self.word_map.items()}

        # PyTorch transformation pipeline for the image
        self.transform = transform
        if self.transform is None:
            self.transform = get_transform(train=(split == 'train'))

    def __getitem__(self, i):
        # Read image
        img_path = self.image_paths[i]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.split == 'train':
            caption = torch.LongTensor(self.captions[i])
            caption_length = torch.LongTensor([self.caption_lengths[i]])
            return img, caption, caption_length
        else:
            # For test, we just need the image ID for creating the submission
            img_id = os.path.basename(img_path).split('.')[0]  # Remove file extension
            return img, img_id

    def __len__(self):
        return len(self.image_paths)

class CaptionTestDataset(Dataset):
    """
    A PyTorch Dataset class for test data without captions.
    """

    def __init__(self, test_folder, transform=None):
        """
        :param test_folder: folder with test images
        :param transform: image transform pipeline
        """
        # Get all image paths
        self.image_paths = []
        self.image_ids = []
        
        for filename in os.listdir(test_folder):
            if filename.endswith('.jpg'):
                self.image_paths.append(os.path.join(test_folder, filename))
                self.image_ids.append(os.path.splitext(filename)[0])  # Remove extension

        # PyTorch transformation pipeline for the image
        self.transform = transform
        if self.transform is None:
            self.transform = get_transform(train=False)

    def __getitem__(self, i):
        # Read image
        img_path = self.image_paths[i]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, self.image_ids[i]

    def __len__(self):
        return len(self.image_paths) 