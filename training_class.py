import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset_x4_y1, Dataset_x1_y1
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os # to check if a plot already exists
from torchsummary import summary
import json

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def train_model(self, dataset, num_epochs, batch_size, learning_rate, model_name):
        # save time:
        tic = time.perf_counter()  # Start time

        # switch to graphic card
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        print("Device = " + device)
        self.to(device)
        self.double()

        #initialize lists to store losses
        train_losses = []
        val_losses = []

        # Datasets:
        shuffle = True
        pin_memory = True
        num_workers = 1

        train_set, val_set = torch.utils.data.random_split(dataset, [math.ceil(len(dataset) * 0.8),
                                                                        math.floor(len(dataset) * 0.2)])
        train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(dataset=val_set, shuffle=shuffle, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Training loop
            self.train()
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for i, (input, target) in enumerate(loop):
                # Forward pass, loss calculation, backward pass, optimization, etc.
                input = input.to(device)
                target = target.to(device)
                outputs = self(input.double())
                loss = criterion(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(trainloss=train_loss/(i+1))


            # calculate average train loss for this epoch
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # write train loss value in the bar
            #loop.set_postfix(trainloss=avg_train_loss)


            # Validation loop
            self.eval()

            # Create a tqdm progress bar for the validation loop

            with torch.no_grad():
                for ind, (input, target) in enumerate(val_loader):
                    input = input.to(device)
                    target = target.to(device)
                    outputs = self(input.double())
                    loss = criterion(outputs, target)
                    val_loss+=loss.item()

            # Calculate the average val losses and add to list
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Save the model after each epoch
            self.save_model(epoch, model_name)



        self.save_loss_plot(model_name, num_epochs, train_losses, val_losses)
        self.save_proc_time(model_name, tic)
        self.save_losses_data(model_name, num_epochs, train_losses, val_losses)



    def save_model(self, epoch, model_name):
        model_dir = os.path.join("models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
        torch.save(self.state_dict(), model_path)

    def save_loss_plot(self, model_name, num_epochs, train_losses, val_losses):
        # Create a list of epochs for the x-axis
        epochs = range(1, num_epochs + 1)

        # Create the losses plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue') #  HTML color names are possible
        plt.plot(epochs, val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.grid(True)

        # Save the losses plot in the same folder as the model
        model_dir = os.path.join("models", model_name)
        losses_plot_filename = os.path.join(model_dir, f"losses_plot_{model_name}.png")
        plt.savefig(losses_plot_filename)

        plt.show()

    def save_proc_time(self, model_name, start_time):
        # Calculate the training process time
        end_time = time.perf_counter()
        proc_time = end_time - start_time

        # Save the training process time to a text file
        model_dir = os.path.join("models", model_name)
        proc_time_filename = os.path.join(model_dir, f"proc_time_{model_name}.txt")

        # Format the time and save it to the file
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(proc_time))
        with open(proc_time_filename, "w") as file:
            file.write(f"Training process duration: {formatted_time}")

    def save_losses_data(self, model_name, epochs, train_losses, val_losses):
        # Create a dictionary to store losses data
        losses_data = {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses
        }


        # Save the losses data as a NumPy .npz file
        model_dir = os.path.join("models", model_name)
        losses_data_filename = os.path.join(model_dir, f"losses_data_{model_name}.npz")
        np.savez(losses_data_filename, **losses_data)

        # save the losses in .txt
        filename = os.path.join(model_dir,'losses_data.txt')

        # Save the dictionary to a text file in JSON format
        with open(filename, 'w') as file:
            json.dump(losses_data, file)

if __name__ == '__main__':
    model = BaseModel()
