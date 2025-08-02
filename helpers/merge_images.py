import os
import shutil
from pathlib import Path


def merge_folders():
    # Define source and destination directories
    generated_dir = Path("../main/Trial Output-1/generated_stable")
    train_dir = Path("../dataset/train")
    output_dir = Path("GAN_train")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get list of folders (1 to 102)
    folders = [str(i) for i in range(1, 103)]

    for folder in folders:
        # Create corresponding output folder
        output_folder = output_dir / folder
        output_folder.mkdir(exist_ok=True)

        # Copy images from generated_stable folder
        generated_folder = generated_dir / folder
        if generated_folder.exists():
            for img_path in generated_folder.glob("*"):
                if img_path.is_file():
                    shutil.copy2(img_path, output_folder / img_path.name)

        # Copy images from train folder
        train_folder = train_dir / folder
        if train_folder.exists():
            for img_path in train_folder.glob("*"):
                if img_path.is_file():
                    shutil.copy2(img_path, output_folder / img_path.name)

        print(f"Merged images for folder {folder}")


if __name__ == "__main__":
    merge_folders()