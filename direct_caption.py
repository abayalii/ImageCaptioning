import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from tqdm import tqdm

# Basic model classes
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use ResNet as the base CNN
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the last fully connected layer (the classification layer)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None, max_len=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # outputs: (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)              # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)      # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

# Simple vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4  # Next available index
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

# Dataset for test images    
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.folder_path, image_file)
        image_id = os.path.splitext(image_file)[0]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_id

def caption_images(test_folder, output_file, embed_size=256, hidden_size=512):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Create a vocabulary (for this direct approach, we'll use a very simple vocab)
    vocab = Vocabulary()
    for word in ['a', 'an', 'the', 'is', 'are', 'on', 'in', 'at', 'with', 'man', 'woman', 
                 'child', 'person', 'dog', 'cat', 'house', 'car', 'building', 'tree', 'sky', 
                 'water', 'beach', 'street', 'city', 'mountain', 'river', 'lake', 'ocean', 
                 'field', 'forest', 'park', 'road', 'walking', 'standing', 'sitting', 'playing',
                 'running', 'eating', 'drinking', 'looking', 'smiling', 'holding', 'wearing',
                 'hat', 'shirt', 'pants', 'shoes', 'dress', 'jacket', 'coat', 'bike', 'bicycle',
                 'bus', 'train', 'boat', 'airplane', 'bird', 'flower', 'fruit', 'table', 'chair',
                 'bed', 'clock', 'book', 'cup', 'plate', 'bowl', 'bottle', 'glass', 'food',
                 'phone', 'computer', 'laptop', 'camera', 'picture', 'photo', 'and', 'of', 'next',
                 'to', 'near', 'around', 'behind', 'front', 'big', 'small', 'young', 'old', 'new',
                 'white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'brown', 'pink',
                 'purple', 'gray', 'sunny', 'cloudy', 'rainy', 'snowy', 'day', 'night', 'morning',
                 'evening', 'afternoon', 'colorful', 'beautiful', 'pretty', 'tall', 'short', 'thin',
                 'thick']:
        vocab.add_word(word)
    
    # Load models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
    
    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Load test images
    test_dataset = TestImageDataset(test_folder, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Generate captions
    results = []
    
    for images, image_ids in tqdm(test_loader, desc='Generating captions'):
        # Prepare image
        images = images.to(device)
        
        # Generate caption
        with torch.no_grad():
            features = encoder(images)
            sampled_ids = decoder.sample(features)
            
        # Convert word_ids to words
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word.get(word_id, '<unk>')
            if word == "<end>":
                break
            if word not in ["<start>", "<pad>"]:
                sampled_caption.append(word)
        
        caption = ' '.join(sampled_caption)
        
        # Add some basic template captions if the generated caption is too short
        if len(sampled_caption) < 3:
            caption = "a person in a room"
        
        # Store result
        results.append({'image_id': image_ids[0], 'caption': caption})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Try to convert image_id to integer if possible
    try:
        df['image_id'] = df['image_id'].astype(int)
    except ValueError:
        print("Warning: Could not convert image_id to integer. Keeping as string.")
    
    df.to_csv(output_file, index=False)
    print(f"Created submission file: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for test images using a simple model')
    parser.add_argument('--test_folder', default='test/test', help='path to test images folder')
    parser.add_argument('--output', default='submission_direct.csv', help='output csv file')
    args = parser.parse_args()
    
    caption_images(args.test_folder, args.output) 