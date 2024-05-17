import os
from models import UNet, UNet_color, UNet_Variational
from dataset import CustomImageDataset
import torch

if __name__ == '__main__':
    print('Okaaay - Let\'s go...')
    device = ("cuda" if torch.cuda.is_available() else "cpu")


    lr = 0.001 # 0.001 for CNN1D3D, 0.0001 for CNN1D
    batch = 32   # open for testing
    epoch = 50
    dropout = 0
    channels = 128*8  #32
    color = 'color'
    finetuning= False #True # a variable that incooperates finetuning


    model_list= [f'UNet_2D_2Blocks_{dropout}dropout_{channels}channels_lr{lr}',
                 f'UNet_2D_VAE_2Layer_{color}_{dropout}dropout_{channels}channels_epoch{epoch}_lr{lr}',
                 f'UNet_2D_2Blocks_{dropout}dropout_{channels}channels_lr{lr}_nopretraining']
    # this is for version 2 'training_class':

    model_name = model_list[2]
    model = UNet_color(d1=256, d2=16, channels=channels, dropout=dropout).to(device)

    # chose Model based on Modelname
    #if model_name == model_list[0] or model_name == model_list[2]:
        #model = UNet_color(d1=256, d2=16, channels=channels, dropout=dropout).to(device)
    #elif model_name ==model_list[1]:
        #model = UNet_Variational(d1=256, d2=16, channels=channels, dropout=dropout).to(device)


    simulationpath = '/beegfs/project/bmbf-need/spectral-analysis/cnn-spectral-analysis/data/Impact_Echo_Machine_Learning/database_autoencoder/new_approach/simulated_set/IE_2D_random_setup_sound/B_scans_rgb/sound'
    finetuningpath = '/beegfs/project/bmbf-need/spectral-analysis/cnn-spectral-analysis/data/Impact_Echo_Machine_Learning/database_autoencoder/new_approach/realworld_DATA/training/sound'


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



    for finetuning in [False,True]:
        if finetuning==True:
            dataset = CustomImageDataset(transform='color',
                                         path=finetuningpath)  # transform = 'gray' for grayscale pictures
            try:
                model.load_state_dict(torch.load(model_path))
            except Exception as e:
                print(f'Unable to load {model_name} from {model_path}. Error: {e}')
            print(f'Model {model_name} states loaded and ready for finetuning')
            model_name = model_name +'finetuned'
            model.train_model(dataset = dataset, num_epochs= epoch,batch_size= batch,
                          learning_rate=lr, model_name=model_name)



        else:
            dataset = CustomImageDataset(transform='color',
                                         path=finetuningpath)  # transform = 'gray' for grayscale pictures
            model.train_model(dataset = dataset, num_epochs= epoch,batch_size= batch,
                          learning_rate=lr, model_name=model_name)


