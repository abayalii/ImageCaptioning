import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

# Define base directory relative to the script location
# Assumes the script is in the project root (ImageCaptioning/)
BASE_DIR = os.getcwd() # Get current working directory, which should be the project root

# Define paths based on the project structure
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test', 'test')
SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.csv')

def display_captions(num_images=5):
    """Reads submission.csv and displays a specified number of images with their captions."""
    
    if not os.path.exists(SUBMISSION_PATH):
        print(f"Error: Submission file not found at {SUBMISSION_PATH}")
        return

    try:
        results_df = pd.read_csv(SUBMISSION_PATH)
    except Exception as e:
        print(f"Error reading submission file {SUBMISSION_PATH}: {e}")
        return

    print(f"Submission file '{SUBMISSION_PATH}' loaded. First {num_images} entries:")
    print(results_df.head(num_images))
    print("\nDisplaying images and captions...")

    # Determine the number of rows and columns for the subplot
    # Aim for a somewhat square layout, e.g., 2x3 for 5 images, 3x3 for 9, etc.
    cols = min(num_images, 5) # Max 5 images per row
    rows = (num_images + cols - 1) // cols # Calculate rows needed

    plt.figure(figsize=(3 * cols, 4 * rows)) # Adjust figure size based on cols/rows

    for i in range(min(num_images, len(results_df))):
        try:
            # Ensure image_id is treated as string for path construction, as in CSV
            img_id = str(results_df['image_id'].iloc[i]) 
            caption = str(results_df['caption'].iloc[i])
        except KeyError as e:
            print(f"Error: Column {e} not found in submission.csv. Please check the file format.")
            return
        except IndexError:
            print(f"Error: Not enough rows in submission.csv to display {num_images} images.")
            break # Stop if we run out of entries
        
        img_path = os.path.join(TEST_IMAGES_DIR, f"{img_id}.jpg") # Assuming .jpg extension
        
        plt.subplot(rows, cols, i + 1)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                plt.imshow(img)
            except Exception as e:
                print(f"Error opening or displaying image {img_path}: {e}")
                plt.text(0.5, 0.5, 'Error loading image', ha='center', va='center')
        else:
            print(f"Warning: Image not found at {img_path}")
            plt.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        plt.title(f"ID: {img_id}", fontsize=10)
        wrapped_caption = textwrap.fill(caption, width=25) # Wrap caption text
        plt.xlabel(wrapped_caption, fontsize=9)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
    print("\nImage and caption review complete.")

if __name__ == "__main__":
    num_to_display = 10 # You can change this number
    print(f"Attempting to display the first {num_to_display} captioned images...")
    display_captions(num_images=num_to_display) 