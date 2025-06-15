import os
from utils import create_input_files

# Define paths and parameters
TRAIN_IMAGE_DIR = "train/train"
TRAIN_CAPTIONS_FILE = "train.csv"
TEST_IMAGE_DIR = "test/test"
TEST_FILE = "test.csv"
OUTPUT_FOLDER = "data_output"
MIN_WORD_FREQ = 5
MAX_LEN = 100
CAPTIONS_PER_IMAGE = 5

if __name__ == "__main__":
    # Check if directories exist
    for directory in [TRAIN_IMAGE_DIR, TEST_IMAGE_DIR]:
        assert os.path.exists(directory), f"Directory {directory} does not exist"
    
    # Check if csv files exist
    for file in [TRAIN_CAPTIONS_FILE, TEST_FILE]:
        assert os.path.exists(file), f"File {file} does not exist"
    
    # Create input files
    create_input_files(
        train_image_dir=TRAIN_IMAGE_DIR,
        train_captions_file=TRAIN_CAPTIONS_FILE,
        test_image_dir=TEST_IMAGE_DIR,
        test_file=TEST_FILE,
        output_folder=OUTPUT_FOLDER,
        min_word_freq=MIN_WORD_FREQ,
        max_len=MAX_LEN,
        captions_per_image=CAPTIONS_PER_IMAGE
    )
    
    print("Input files created successfully!") 