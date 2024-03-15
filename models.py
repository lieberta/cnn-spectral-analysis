import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # for summarizing the model
import math
from training_class import BaseModel

class EncoderBlock(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, out_c, dropout=0):
        super(EncoderBlock, self).__init__()
        self.dropout_prob = dropout
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        #self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) ######################

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        #x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        return x
class DecoderBlock(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, out_c,dropout =0, padding = (0,0,0)):
        super(DecoderBlock, self).__init__()
        self.dropout_prob = dropout

        self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size= 2,stride=2, output_padding = padding)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1) # here additional input_channels for cross connections can be added
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.up(x)
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, d1 = 256, d2 = 16, channels=64, dropout=0):
        super(UNet, self).__init__()
        # initialize the latent space dimensions:
        self.d1 = math.floor(math.floor(d1/2)/2)
        self.d2 = math.floor(math.floor(d2/2)/2)
        #   self.input_dim = self.d1*self.d2

        # initialize how much outputchannels the layers should have
        self.channel_parameter = channels

        # Encoder
        self.encoder1 = EncoderBlock(in_c = 1, out_c = self.channel_parameter,dropout = dropout)
        self.encoder2 = EncoderBlock(in_c = self.channel_parameter, out_c = 2*self.channel_parameter, dropout = dropout)

        # Decoder
        self.dblock1=DecoderBlock(2*self.channel_parameter, self.channel_parameter,dropout = dropout,padding=(0,0))    # additional 0 channels for the crossconnection
        self.dblock2=DecoderBlock(self.channel_parameter,1,dropout = dropout,padding=(0,0))    # additional 0 channels for the crossconnection


    def forward(self, x):


        x= self.encoder1(x)
        x= self.encoder2(x)
                #print(f'after second encoding: {x_t[0].shape}')


        # Decoder:
        x = self.dblock1(x)
        x = self.dblock2(x)

        return x

class UNet_color(UNet):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, d1 = 256, d2 = 16, channels=64, dropout=0):
        super(UNet_color, self).__init__(d1, d2, channels, dropout)

        # override the channels
        self.encoder1 = EncoderBlock(in_c = 3, out_c = self.channel_parameter,dropout = dropout)
        self.dblock2=DecoderBlock(self.channel_parameter,3,dropout = dropout,padding=(0,0))    # additional 0 channels for the crossconnection



class UNet_Variational(UNet_color):
    def __init__(self, d1 = 256, d2 = 16, channels=64, dropout=0):
        super(UNet_Variational, self).__init__(d1, d2, channels, dropout)

        # Middle mu:
        self.conv1mu = nn.Conv3d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1mu = nn.GroupNorm(num_groups=int(64/8), num_channels=64)
        self.conv2mu = nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2mu = nn.GroupNorm(num_groups=int(64/8), num_channels=64)
        # Middle logvar:
        self.conv1logvar = nn.Conv3d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn1logvar = nn.GroupNorm(num_groups=int(64/8), num_channels= 64) #nn.BatchNorm3d(out_c)
        self.conv2logvar = nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.gn2logvar = nn.GroupNorm(num_groups=int(64/8), num_channels=64)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        # Encoder steps
        x = self.encoder1(x)
        x = self.encoder2(x)

        # Apply the new layers here, between the encoder and decoder
        # Middle Variational:
        mu = self.conv1mu(self.gn1mu(self.conv2mu(self.gn2mu(x))))
        logvar = self.conv1logvar(self.gn1logvar(self.conv2logvar(self.gn2logvar(x))))

        x = reparameterize(mu, logvar)

        # Decoder steps
        x = self.dblock1(x)
        x = self.dblock2(x)

        return x, mu, logvar




if __name__ == '__main__':
    # work in progress on UNet_timeconv
    print(torch.cuda.is_available())

    model = UNet_color().cuda() #UNet().cuda()


    x = torch.arange(3*16*256).reshape(1, 3, 256,16).cuda()#(1*16*256).reshape(1, 1, 256,16).cuda()


    #btensor = batchnorm(x.float())

    y = model(x.float())
    print(y.shape)



    #model2 = CNN1D3D().cuda()
    #summary(model, (4, 1, 61, 81, 31))


