import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import Dataset_x4_y1
from models import UNet_timeconv, UNet_timeconv_cross, UNet_timeconv_deviation
import os
from torch.utils.data import DataLoader
import torch
# create the whole visualization:
def visualize_experiment(experiment):
    min = 19.998600006103516
    dist = 1344.7513999938965
    max = 600

    def denormalize(tensor):
        tensor_denorm = tensor * dist + min
        return tensor_denorm

    def create_slices(x):
        with torch.no_grad():

            output_temp = []
            target_temp = []

            dataiter = iter(data_loader)
            sample_batch = dataiter.next()
            print(f'targetshape of dataloader: {sample_batch[1].shape} \n length: {len(data_loader.dataset)}')

            j = 0
            for i, (input, target) in enumerate(data_loader):
                # the amount of (input,target) packages in a single experiment is here n_timesteps-input_timesteps
                # therefore for each experiment, take the (input, target) pairs out of the right thing
                if i in range(experiment * (n_timesteps - input_timesteps),
                              (experiment + 1) * (n_timesteps - input_timesteps)):
                    j += 1

                    input = input.to(device)
                    input = input.double()

                    output = model(input).to(device)
                    output = denormalize(output)
                    target = denormalize(target)
                    target_temp.append(target[0, 0, x, :, :].cpu().numpy())
                    output_temp.append(output[0, 0, x, :, :].cpu().numpy())

                    if j == n_timesteps - 4:
                        break

            return target_temp, output_temp

    # Create a function to generate combined plots
    def create_combined_plot(dataset, predictions, t):
        plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
        plt.subplot(1, 2, 1)
        plt.imshow(dataset[t][:, :].T, cmap='coolwarm', vmin=min, vmax=max)
        plt.colorbar()
        plt.title(f'Dataset - Time Step {t}')

        plt.subplot(1, 2, 2)
        # plt.imshow(predictions[t,0, :, :], cmap='coolwarm', vmin=min, vmax=max)
        plt.imshow(predictions[t][:, :].T, cmap='coolwarm', vmin=min, vmax=max)
        plt.colorbar()
        plt.title(f'Model Prediction - Time Step {t}')

        plt.savefig(video_dir + f'/combined_frame_{t:04d}.png')
        plt.close()

    video_dir = os.path.join(model_dir, f"video{experiment}")
    # creates folder if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)

    # Assuming dataset and predictions are your 3D temperature arrays
    # Adjust min_value, max_value, frame_width, frame_height, and other parameters

    # creates targets and predictions for a slice with fixed x

    print(f'Create slices...')
    target_temp, output_temp = create_slices(x=15)

    print(f'Length of target list = {len(target_temp)}')
    print(f'shape of one target: {target_temp[0].shape}')

    print('Create combined frames and save .png')
    # Loop to create combined frames
    for t in range(n_timesteps - 4):
        create_combined_plot(target_temp, output_temp, t)

    print('Create final video...')
    # Set the output video file name and frame rate
    video_output = video_dir + '/temperature_video.mp4'
    frame_rate = 1  # Adjust the frame rate as needed

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    frame_width = 2400
    frame_height = 400
    video_writer = cv2.VideoWriter(video_output, fourcc, frame_rate, (frame_width, frame_height))

    # Loop through frames and add them to the video
    for t in range(n_timesteps - 4):
        frame = cv2.imread(video_dir + f'/combined_frame_{t:04d}.png')
        video_writer.write(frame)

    # Release the VideoWriter and clean up
    video_writer.release()
    cv2.destroyAllWindows()

# experiment = 0 # number of experiment
n_timesteps = 60 # number of timesteps
input_timesteps = 4 # number of input timesteps for CNN1D3D

device = ("cuda" if torch.cuda.is_available() else "cpu")

model_name = f'UNet_3D_1D_2Layer_deviation_0dropout_0sideconnections_epoch50_lr0.001'
dataset = Dataset_x4_y1()
model = UNet_timeconv_deviation(d1=61, d2=81, d3=31).double().to(device)

model_dir = os.path.join("models", model_name)

model.load_state_dict(torch.load(model_dir+'/epoch_49.pth'))



print('Load Dataset...')
dataset = Dataset_x4_y1()
data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)

for i in range(5):
    visualize_experiment(i)



