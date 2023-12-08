import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, path = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans/sound"):
        #self.image_dir = "./data/database_autoencoder/IE_2D_random_setup_sound/B_scans/sound"
        self.image_dir = path
        self.image_files = self.get_all_image_files(self.image_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

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
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image

def tensor_to_picture(tensor):
    image_np = tensor.numpy()
    image_np = image_np.reshape((256, 16, 1))
    plt.imshow(image_np, cmap='gray')  # Use cmap='gray' if your image is grayscale
    plt.show()



if __name__ == '__main__':

    Set = CustomImageDataset()
    for x, y in enumerate(Set):
        print (x,y)
        break

    print (y.size()) # [1, 256, 16]

    tensor_to_picture(y)

# convert tensor to picture:
