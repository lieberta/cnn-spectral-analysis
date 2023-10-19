import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # for summarizing the model
import math
from training_class import BaseModel
class EncoderBlock_cross(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, out_c, dropout=0.5):
        super(EncoderBlock_cross, self).__init__()
        self.dropout_prob = dropout
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        y = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(y)
        return x, y
class DecoderBlock_cross(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c, cross_c,out_c,dropout =0.5, padding = (0,0,0)):
        super(DecoderBlock_cross, self).__init__()
        self.dropout_prob = dropout

        self.up = nn.ConvTranspose3d(in_c, in_c, kernel_size= 2,stride=2, output_padding = padding)
        self.conv1 = nn.Conv3d(in_c+cross_c, out_c, kernel_size=3, padding=1) # here additional input_channels for cross connections can be added
        self.bn1 = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x,x_cross):
        x = self.up(x)
        x = torch.cat((x,x_cross), dim = -4)
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
class UNet(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, d1 = 61, d2 = 81, d3 = 31):
        super(UNet, self).__init__()
        # initialize the latent space dimensions:
        self.d1 = math.floor(math.floor(d1/2)/2)
        self.d2 = math.floor(math.floor(d2/2)/2)
        self.d3 = math.floor(math.floor(d3/2)/2)
        self.input_dim = self.d1*self.d2*self.d3
        # Encoder
        self.encoder1 = EncoderBlock_cross(in_c = 1, out_c = 8,dropout = 0)
        self.encoder2 = EncoderBlock_cross(in_c = 8, out_c = 16, dropout = 0)
        self.tempencoding = nn.Conv3d(in_channels=16, out_channels=16 * 4, kernel_size=(4, 1, 1), stride=1)
        # Decoder
        self.dblock1=DecoderBlock_cross(16*4, 16,8,padding=(0,0,1))    # additional 16 channels for the crossconnection
        self.dblock2=DecoderBlock_cross(8,8,1,padding=(1,1,1))    # additional 8 channels for the crossconnection

    def forward(self, x):
        # Spatial Encoding:
        x_t = torch.chunk(x, 4, dim=1)  # splits x into a tuple of 4 [1,1,61,81,31] tensors along the time dimension
        x_t = list(x_t) # convert x_t to a list so i can modify the elements


        for i in range(4):         # generates for each timestep a squeezed tensor of [1,61,81,31] and spatially encodes all tensors
            x_t[i] = torch.squeeze(x_t[i],1)
            x_t[i], x_cross1 = self.encoder1(x_t[i]) # x_cross1 shape: [batch, 8, 61,81,31]
            x_t[i], x_cross2 = self.encoder2(x_t[i])    # x_cross2 shape [batch, 16, 61,81,31]
            #if i ==0:
                #print(f'after second encoding: {x_t[0].shape}')
        x = torch.cat([tensor.unsqueeze(-5) for tensor in x_t], dim=-5) # concats along the time to [batch,time, channels ,x,y,z]
        #print(f'here {x.shape}')
        # Temporal Encoding:
        x = x.view(x.shape[0], 16, 4, 1, self.input_dim)  # reshape the dimensions such that a 3D layer can process the time
        x = self.tempencoding(x) # reduces the time dimension to 1
        x = x.view(x.shape[0], -1, self.d1, self.d2, self.d3) # reshapes the tensor to [batch, output_channels, x,y,z]

        # Decoder:
        x = self.dblock1(x,x_cross2)
        x = self.dblock2(x,x_cross1)

        return x

class Conv(nn.Module):
    # a convolution that reshapes the output of an 3DCNN
    def __init__(self):
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(4,1), stride = 1)
    def forward(self, x):

        x = x.reshape(-1,1,4,61*81*31)     #  batch,4,1,x,y,z
        x = self.conv(x)
        x = x.reshape (-1,1,61,81,31)
        return x

if __name__ == '__main__':
    # work in progress on UNet_timeconv

    #model = UNetLSTM().cuda()
    model = UNet().cuda()
    #summary(model, (4, 1, 61, 81, 31))
    x = torch.arange(1*4*61*81*31).reshape(1, 4, 1, 61, 81, 31).cuda() #(batch, timesteps,input_channels, 61,81,31)
    y = model(x.float())
    print(y.shape)

    #model2 = CNN1D3D().cuda()
    #summary(model, (4, 1, 61, 81, 31))


