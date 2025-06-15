import os
import argparse
import subprocess
import torch

def run_pipeline(epochs=5, batch_size=32, beam_size=3):
    """
    Run the entire image captioning pipeline - data preparation, training, and caption generation.
    
    :param epochs: number of epochs to train for
    :param batch_size: batch size for training
    :param beam_size: beam size for caption generation
    """
    # Define paths
    data_folder = "data_output"
    output_folder = "output_models"
    
    # 1. Prepare data
    print("\n=== Step 1: Preparing data ===")
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print("Running data preparation...")
        subprocess.run(["python", "create_input_files.py"])
    else:
        print(f"Data folder {data_folder} already exists. Skipping data preparation.")
    
    # Update train.py with the specified epochs and batch size
    with open('train.py', 'r') as f:
        train_content = f.read()
    
    train_content = train_content.replace('epochs = 10', f'epochs = {epochs}')
    train_content = train_content.replace('batch_size = 32', f'batch_size = {batch_size}')
    
    with open('train.py', 'w') as f:
        f.write(train_content)
    
    # 2. Train model
    print("\n=== Step 2: Training model ===")
    print(f"Training for {epochs} epochs with batch size {batch_size}...")
    subprocess.run(["python", "train.py"])
    
    # 3. Generate captions
    print("\n=== Step 3: Generating captions ===")
    
    # Find the last checkpoint file
    checkpoints = [f for f in os.listdir(output_folder) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoint files found. Training may have failed.")
        return
    
    # Sort by epoch number and get the latest
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))[-1]
    checkpoint_path = os.path.join(output_folder, latest_checkpoint)
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Generating captions with beam size {beam_size}...")
    
    subprocess.run([
        "python", "caption_from_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--word_map", os.path.join(data_folder, "word_map.json"),
        "--test_folder", "test/test",
        "--output", "submission.csv",
        "--beam_size", str(beam_size)
    ])
    
    print("\n=== Pipeline completed ===")
    print("Submission file created: submission.csv")

def main():
    parser = argparse.ArgumentParser(description='Run the entire image captioning pipeline.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for caption generation')
    args = parser.parse_args()
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU. Training will be slow.")
        if args.batch_size > 16:
            print(f"Reducing batch size from {args.batch_size} to 16 for CPU training.")
            args.batch_size = 16
    
    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        beam_size=args.beam_size
    )

if __name__ == '__main__':
    main() 