import numpy as np
import matplotlib.pyplot as plt
#import cv2
from dataset import CustomImageDataset
from models import UNet
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
# create the whole visualization:


device = ("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


dataset = CustomImageDataset()


channels= 8
model = UNet(d1=256,d2=16,channels=channels).to(device)


model_name =[
    f'UNet_2D_2Layer_0dropout_{channels}channels_epoch50_lr0.001'

]
i = 0 # chose one Modelname out of the list


model_folder = f'./models/{model_name[i]}'
model_path = f'{model_folder}/{model_name[i]}.pth'

model.load_state_dict(torch.load(model_path))
model.eval()


anomaly_path = "./data/database_autoencoder/IE_2D_random_setup_honeycomb/B_scans/defect"

test_path ="./data/database_autoencoder/IE_2D_random_setup_sound/B_scans/testset"


test_set = DataLoader(dataset=CustomImageDataset(path = test_path), shuffle=False, batch_size=1)
anomaly_set = DataLoader(dataset=CustomImageDataset(path = anomaly_path), shuffle=False, batch_size=1)

def loss_list(set):
    loss_set = []
    for i, x in enumerate(set):
            y = model(x)
            loss = criterion(x,y).item()
            loss_set.append(loss)
    return loss_set


test_losses = loss_list(test_set)
anomaly_losses = loss_list(anomaly_set)


# Create a range of numbers equal to the length of the loss lists
x_range = range(1, len(test_losses) + 1)

# cut the anomaly loss list to the same
anomaly_losses = anomaly_losses[:len(x_range)]


# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x_range, test_losses, 'g', label='Test Losses')  # 'g' is for green color
plt.plot(x_range, anomaly_losses, 'r', label='Anomaly Losses')  # 'r' is for red color

plt.xlabel('Sample')
plt.ylabel('Loss')
plt.title(f'Loss Comparison {model_name[i]}')
plt.legend()

# Specify the folder name
detection_folder = f'{model_folder}/detection_plot'
# Create the folder if it doesn't exist
os.makedirs(detection_folder, exist_ok=True)
# Specify the file name
file_name = f"loss_comparison_{model_name[i]}.png"
# Full path for saving
save_path = os.path.join(detection_folder, file_name)
# Save the plot
plt.savefig(save_path)

plt.show()


