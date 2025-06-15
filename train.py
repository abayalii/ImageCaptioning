import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import clip_gradient
import os
import json

# Data parameters
data_folder = 'data_output'  # folder with data files saved by create_input_files.py
output_folder = 'output_models'  # folder to save models

# Model parameters
embed_dim = 512  # embedding dimension
attention_dim = 512  # attention network dimension
decoder_dim = 512  # decoder dimension
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training parameters
start_epoch = 0
epochs = 3  # number of epochs to train for
batch_size = 16
workers = 4  # number of workers for data loading
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # gradient clip value (to prevent exploding gradients)
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder
checkpoint = None  # path to checkpoint, None if none

def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_folder, word_map

    # Create directories if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load word map (word2ix)
    with open(os.path.join(data_folder, 'word_map.json'), 'r', encoding='utf-8') as j:
        word_map = json.load(j)
    
    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder()
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                      embed_dim=embed_dim,
                                      decoder_dim=decoder_dim,
                                      vocab_size=len(word_map),
                                      dropout=dropout)
        
        encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=encoder_lr) if fine_tune_encoder else None
        decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=decoder_lr)
    else:
        # Load checkpoint
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']
        best_bleu4 = checkpoint['bleu-4']
        
    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = CaptionDataset(data_folder, 'train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=True
    )
    
    # Train the model
    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train(train_loader=train_loader,
             encoder=encoder,
             decoder=decoder,
             criterion=criterion,
             encoder_optimizer=encoder_optimizer,
             decoder_optimizer=decoder_optimizer,
             epoch=epoch)
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'encoder': encoder,
            'decoder': decoder,
            'encoder_optimizer': encoder_optimizer,
            'decoder_optimizer': decoder_optimizer,
            'bleu-4': best_bleu4
        }
        
        filename = f'checkpoint_epoch_{epoch}.pth'
        
        if fine_tune_encoder:
            filename = f'checkpoint_epoch_{epoch}_fine_tuned.pth'
        
        torch.save(state, os.path.join(output_folder, filename))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss function
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


from torch.nn.utils.rnn import pack_padded_sequence

if __name__ == '__main__':
    main() 