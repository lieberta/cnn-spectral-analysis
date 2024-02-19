import os
import re
import shutil

def get_all_image_files_and_copy(dir, target_dir="./data/database_autoencoder/IE_2D_random_setup_sound/B_scans_rgb_ny600/sound"):
    """Recursively fetch all image file paths from a directory and its subdirectories and copy files with 'ny' value over 600."""
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extracting the number after 'ny' using regular expressions
                match = re.search(r'ny(\d+)', file)
                if match:
                    ny_number = int(match.group(1))
                    if ny_number > 600:
                        # Construct the full source path
                        source_path = os.path.join(root, file)
                        # Construct the target path
                        target_path = os.path.join(target_dir, os.path.relpath(root, dir), file)
                        # Make sure the target directory exists
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        # Copy the file to the target directory
                        shutil.copy2(source_path, target_path)

# Example usage
# Assuming the class and method definition above,
# create an instance of the class and call the method like this:
path = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans_rgb/sound"
get_all_image_files_and_copy(dir = path)