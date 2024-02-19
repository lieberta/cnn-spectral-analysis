import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, path = None, transform = None):

        if path ==None and transform == 'gray':
            self.image_dir = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans/sound"

        elif path == None and transform == 'color':
            self.image_dir = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans_rgb/sound"

        elif path == None and transform == 'color_ny600':
            self.image_dir = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans_rgb_ny600/sound"

        elif path != None: # if you additionally include a path to the initialization then it overrides the train-setpath
            self.image_dir = path


        self.image_files = self.get_all_image_files(self.image_dir)

        # Transform selection
        if transform == 'gray':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        elif transform in ['color', 'color_ny600']:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            # Fallback or default transformation
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def get_all_image_files(self, dir):
        """Recursively fetch all image file paths from a directory and its subdirectories."""
        image_files = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        return image


class ModifiedImageDataset(CustomImageDataset):
    def __init__(self, path, transform=None):
        super().__init__(path)  # Call the constructor of the parent class
        # Override the transform if a new one is provided
        if transform is not None:
            self.transform = transform

def tensor_to_picture(tensor):
    image_np = tensor.numpy()
    image_np = image_np.transpose(1, 2, 0)
    plt.imshow(image_np)  # Use cmap='gray' if your image is grayscale
    plt.show()



if __name__ == '__main__':

    Set = CustomImageDataset(transform = 'color')
    for x, y in enumerate(Set):
        print (x,y)
        break

    print (y.size()) # [1, 256, 16]

    tensor_to_picture(y)

# convert tensor to picture:
