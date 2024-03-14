
import os
from models import UNet, UNet_color
from dataset import CustomImageDataset
import torch

if __name__ == '__main__':
    print('Okaaay - Let\'s go...')
    device = ("cuda" if torch.cuda.is_available() else "cpu")


    lr = 0.001 # 0.001 for CNN1D3D, 0.0001 for CNN1D
    batch = 8   # open for testing
    epochs = 100
    dropout = 0
    channels = 128*8 #4 #32
    color = 'color'

    # this is for version 2 'training_class':
    model_name = f'UNet_2D_newtrainset_2Layer_{color}_{dropout}dropout_{channels}channels_epoch{epochs}_lr{lr}'

    dataset = CustomImageDataset(transform = 'color', path='/beegfs/project/bmbf-need/spectral-analysis/cnn-spectral-analysis/data/Impact_Echo_Machine_Learning_2/database_autoencoder/new_approach/IE_2D_random_setup_sound/B_scans_rgb/sound') # transform = 'gray' for grayscale pictures
    model = UNet_color(d1=256, d2=16,channels=channels, dropout=dropout).to(device)



    # Define your model path based on the model name
    model_dir = os.path.join("models", model_name)
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    # Check if the model directory and the specific model file exist
    if os.path.exists(model_dir) and os.path.isfile(model_path):
        # Load the model
        model.load_state_dict(torch.load(model_path))
        print(f"The model '{model_name}' already exists and will be trained further.")
    else:
        # If the directory or model file doesn't exist, print this message
        print(f"A new folder and model '{model_name}' will be created.")
    model.train_model(dataset = dataset, num_epochs= epochs,batch_size= batch,
                      learning_rate=lr, model_name=model_name)



    #def train_model(self, dataset, num_epochs=50, batch_size=64, learning_rate, model_name)
    #model_type = 'CNN3D' # modeltypes: CNN3D, CNN1D3D
    #history, model = train(lr, batch, epochs, model_type)