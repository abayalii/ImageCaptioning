import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
# import torchvision.transforms as transforms # Not strictly needed for BLIP fine-tuning if processor handles it
# import torchvision.models as models # Not needed for BLIP
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer # Kept for now, though BLIP uses its own processor
from transformers import BlipProcessor, BlipForConditionalGeneration # Removed AdamW from here
from tqdm import tqdm

# Define constants
TRAIN_IMAGE_DIR = "train/train"
TRAIN_CSV_FILE = "train.csv"
TEST_IMAGE_DIR = "test/test"
TEST_CSV_FILE = "test.csv"
SUBMISSION_FILE = "submission.csv"
FINETUNED_MODEL_SAVE_PATH = "blip_finetuned_model"

# Fine-tuning parameters
FT_NUM_EPOCHS = 3 # Increased from 1 to 3 for more thorough fine-tuning
FT_BATCH_SIZE = 2  # Reduced from 4 to 2 to address OOM during fine-tuning
FT_LEARNING_RATE = 5e-5 # Common learning rate for fine-tuning transformers

# Prediction parameters (can be different, e.g., if using a different batch size for inference)
PRED_BATCH_SIZE = 8 # Batch size for prediction with BLIP
MAX_CAPTION_LENGTH = 77 # BLIP default max_length for captions is often around 77 for generation

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset for BLIP Fine-tuning
class BlipFineTuningDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        try:
            self.image_ids = self.data['image_id'].astype(str).tolist()
            self.captions = self.data['caption'].astype(str).tolist()
        except KeyError as e:
            print(f"Error: Column {e} not found in {csv_file}. Please check the CSV file.")
            self.image_ids = []
            self.captions = []
        except Exception as e:
            print(f"Error reading data from {csv_file}: {e}")
            self.image_ids = []
            self.captions = []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        caption_text = self.captions[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(f"Warning (Fine-tune Dataset): Could not open image {img_path} (Error: {e}). Returning None.")
            return None, None # Skip this sample if image can't be loaded
        
        return image, caption_text

# Collate function for fine-tuning BLIP
def create_finetune_collate_fn(processor, max_target_length=MAX_CAPTION_LENGTH):
    def collate_fn_finetune(batch):
        # Filter out None samples (where image loading might have failed)
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        if not batch:
            return None # Return None if batch is empty after filtering
            
        images, texts = zip(*batch)
        
        # Process batch using BlipProcessor
        # The processor handles image preprocessing and text tokenization.
        # For fine-tuning captioning, text inputs are the target captions.
        inputs = processor(images=list(images), text=list(texts), 
                           return_tensors="pt", padding="max_length", 
                           truncation=True, max_length=max_target_length)
        
        # The `input_ids` from the processor (for the text) will serve as `labels` for the model.
        inputs['labels'] = inputs['input_ids'].clone()
        # According to BLIP, for loss calculation, label tokens are overridden with -100 for pad tokens
        inputs['labels'][inputs['labels'] == processor.tokenizer.pad_token_id] = -100
        
        return inputs
    return collate_fn_finetune

# Dataset for BLIP Prediction (Test set)
class BlipPredictionDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        try:
            self.image_ids = self.data['image_id'].astype(str).tolist()
        except KeyError:
            print(f"Error: 'image_id' column not found in {csv_file}.")
            self.image_ids = []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        # Image loading will happen in the prediction loop or a prediction collate_fn if batching predictions
        return img_id, img_path


# Main script
if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")

    # --- 1. Fine-tune BLIP model ---
    print("\n--- Starting BLIP Fine-tuning Phase ---")
    try:
        print("Loading BLIP model and processor for fine-tuning...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)
        print("BLIP model and processor loaded.")
    except Exception as e:
        print(f"Error loading BLIP model for fine-tuning: {e}")
        exit()

    fine_tune_dataset = BlipFineTuningDataset(csv_file=TRAIN_CSV_FILE, image_dir=TRAIN_IMAGE_DIR)
    if len(fine_tune_dataset) == 0:
        print("Fine-tuning dataset is empty. Exiting.")
        exit()
    
    collate_fn_finetune = create_finetune_collate_fn(processor, max_target_length=MAX_CAPTION_LENGTH)
    fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=FT_BATCH_SIZE, 
                                      shuffle=True, collate_fn=collate_fn_finetune)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LEARNING_RATE)

    model.train() # Set model to training mode
    for epoch in range(FT_NUM_EPOCHS):
        print(f"\nStarting fine-tuning Epoch {epoch + 1}/{FT_NUM_EPOCHS}")
        epoch_loss = 0
        progress_bar = tqdm(fine_tune_dataloader, desc=f"Fine-tuning Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: # Skip if collate_fn returned None (e.g. all images in batch failed to load)
                print(f"Skipping empty batch {batch_idx + 1}")
                continue

            inputs = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**inputs) # Model returns loss when labels are provided
            loss = outputs.loss
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            else:
                print(f"Warning: Loss is None for batch {batch_idx+1}. Check model inputs and labels.")
        
        avg_epoch_loss = epoch_loss / len(fine_tune_dataloader) if len(fine_tune_dataloader) > 0 else 0
        print(f"Fine-tuning Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    # Save the fine-tuned model
    try:
        model.save_pretrained(FINETUNED_MODEL_SAVE_PATH)
        processor.save_pretrained(FINETUNED_MODEL_SAVE_PATH) # Also save the processor
        print(f"Fine-tuned BLIP model and processor saved to {FINETUNED_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving fine-tuned model: {e}")

    # --- 2. Generate Captions with Fine-tuned BLIP model ---
    print("\n--- Starting Caption Generation with Fine-tuned BLIP Model ---")
    # It's good practice to reload the model and processor for inference,
    # or ensure model is in eval mode and no gradients are computed.
    try:
        print(f"Loading fine-tuned BLIP model and processor from {FINETUNED_MODEL_SAVE_PATH}...")
        # Load the fine-tuned model and processor
        processor = BlipProcessor.from_pretrained(FINETUNED_MODEL_SAVE_PATH)
        model = BlipForConditionalGeneration.from_pretrained(FINETUNED_MODEL_SAVE_PATH)
        model.to(device)
        model.eval() # Set to evaluation mode
        print("Fine-tuned BLIP model and processor loaded.")
    except Exception as e:
        print(f"Error loading fine-tuned BLIP model: {e}. Make sure the path is correct and model was saved.")
        exit()

    pred_dataset = BlipPredictionDataset(csv_file=TEST_CSV_FILE, image_dir=TEST_IMAGE_DIR)
    # For prediction, DataLoader can have batch_size > 1 if BLIP processor and model.generate handle batches
    # For simplicity here, keeping batch_size=1 for prediction loop, processing images individually.
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

    results = []
    if len(pred_dataset) == 0:
        print("Prediction dataset is empty. Cannot generate submission.")
    else:
        progress_bar_pred = tqdm(pred_dataloader, desc="Generating Captions (Fine-tuned BLIP)")
        for img_id_tuple, img_path_tuple in progress_bar_pred: # DataLoader returns tuples for batch_size=1
            img_id_str = img_id_tuple[0]
            img_path = img_path_tuple[0]
            caption = "placeholder caption - error or not found"
            try:
                if not os.path.exists(img_path):
                    # print(f"Warning (Prediction): Image file not found at '{img_path}'. Skipping ID {img_id_str}.")
                    caption = "placeholder caption - image not found"
                else:
                    raw_image = Image.open(img_path).convert('RGB')
                    inputs = processor(images=raw_image, return_tensors="pt").to(device)
                    
                    with torch.no_grad(): # Ensure no gradients are computed during generation
                        generated_output = model.generate(**inputs, max_length=MAX_CAPTION_LENGTH) 
                    
                    caption = processor.decode(generated_output[0], skip_special_tokens=True)

            except Exception as e:
                print(f"Error generating caption for image_id {img_id_str} at path {img_path}: {e}")
                # caption remains the default error placeholder
            
            results.append({"image_id": img_id_str, "caption": caption})

    # Create submission.csv
    if results:
        submission_df = pd.DataFrame(results)
        try:
            submission_df['image_id'] = submission_df['image_id'].astype(int)
        except ValueError:
            print("Warning: Could not convert image_id in submission to int. Keeping as string.")
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission file created with fine-tuned model: {SUBMISSION_FILE}")
    else:
        print("No results to write to submission file.")

    print("Script finished.") 