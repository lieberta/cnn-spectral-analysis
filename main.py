#from training import train
from models import UNet
from dataset import CustomImageDataset
import torch

if __name__ == '__main__':
    print('Okaaay - Let\'s go...')
    device = ("cuda" if torch.cuda.is_available() else "cpu")


    lr = 0.001 # 0.001 for CNN1D3D, 0.0001 for CNN1D
    batch = 8   # open for testing
    epochs = 50
    dropout = 0.2
    channels = 4
    # this is for version 2 'training_class':
    model_name = f'UNet_2D_2Layer_{dropout}dropout_{channels}channels_epoch{epochs}_lr{lr}'
    dataset = CustomImageDataset()
    model = UNet(d1=256, d2=16,channels=channels, dropout=dropout).to(device)
    model.train_model(dataset = dataset, num_epochs= epochs,batch_size= batch,
                      learning_rate=lr,model_name=model_name)



    #def train_model(self, dataset, num_epochs=50, batch_size=64, learning_rate, model_name)
    #model_type = 'CNN3D' # modeltypes: CNN3D, CNN1D3D
    #history, model = train(lr, batch, epochs, model_type)