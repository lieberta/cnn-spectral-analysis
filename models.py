import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary    # for summarizing the model
import math
from training_class import BaseModel

class CNN3D_encoder(nn.Module):
    def __init__(self):
        super(CNN3D_encoder,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 4, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding = 2, padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        return x
class CNN3D_decoder(nn.Module):
        def __init__(self, timesteps):
            super(CNN3D_decoder, self).__init__()
            self.factor = 3
            self.ts = timesteps * self.factor
            self.deconv1 = nn.ConvTranspose3d(in_channels=self.ts * 32, out_channels=self.ts * 16,
                                              kernel_size=(2, 2, 2), stride=2,
                                              padding=0, output_padding=(1, 0, 1))
            self.deconv2 = nn.ConvTranspose3d(in_channels=self.ts * 16, out_channels=self.ts * 8, kernel_size=(2, 2, 2),
                                              stride=2,
                                              padding=2, output_padding=(0, 0, 1))
            self.deconv3 = nn.ConvTranspose3d(in_channels=self.ts * 8, out_channels=1, kernel_size=3, stride=2,
                                              padding=4,
                                              output_padding=0)

        def forward(self, x):
            x = self.deconv1(x)
            x = F.relu(x)
            x = self.deconv2(x)
            x = F.relu(x)
            x = self.deconv3(x)
            x = F.relu(x)

            return x
class CNN3D1D(nn.Module):
    def __init__(self):
        super(CNN3D1D, self).__init__()
        self.factor = 3 #factor for scaling up outputchannels in time convolution
        self.block1 = CNN3D_encoder()
        self.block2 = CNN3D_encoder()
        self.block3 = CNN3D_encoder()
        self.block4 = CNN3D_encoder()
        # here paper flattens the output and uses a fc layer

        self.convfake1d = nn.Conv3d(in_channels=32, out_channels=32 * 4 * self.factor, kernel_size=(4, 1, 1), stride=1)

        self.block_deconv = CNN3D_decoder(4)

    def forward(self, x):
        x1 = x[:, 0, :, :, :, :]  # dimension [batch, in_channel = 1, 61,81,31]
        x2 = x[:, 1, :, :, :, :]
        x3 = x[:, 2, :, :, :, :]
        x4 = x[:, 3, :, :, :, :]

        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        x4 = self.block4(x4)

        x = torch.stack((x1, x2, x3, x4), dim=1, out=None)  # stack them bois together to one fat tensor
        x = x.reshape(-1, 32, 4, 1, 9 * 12 * 5) #dimensions of last cnn layer
        x = self.convfake1d(x)
        x = x.reshape(-1, 32 * 4 * self.factor, 9, 12, 5)
        x = self.block_deconv(x)

        return x

class EncoderBlock(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c,out_c,dropout =0.5):
        super(EncoderBlock, self).__init__()
        self.dropout_prob = dropout
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        return x
class DecoderBlock(nn.Module):
    # one 3D convolutional block with batchnorm, relu activation and two convolution
    # one convolution extends the output channels and one keeps the channels
    def __init__(self, in_c,out_c,dropout =0.5, padding = (0,0,0)):
        super(DecoderBlock, self).__init__()
        self.dropout_prob = dropout

        self.up = nn.ConvTranspose3d(in_c, in_c, kernel_size= 2,stride=2, output_padding = padding)
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1) # here additional input_channels for cross connections can be added
        self.bn1 = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.up(x)
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet_timeconv(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # work in progress
    def __init__(self, d1 = 61, d2 = 81, d3 = 31):
        super(UNet_timeconv, self).__init__()
        # initialize the latent space dimensions:
        self.d1 = math.floor(math.floor(d1/2)/2)
        self.d2 = math.floor(math.floor(d2/2)/2)
        self.d3 = math.floor(math.floor(d3/2)/2)
        self.input_dim = self.d1*self.d2*self.d3
        # Encoder
        self.encoder1 = EncoderBlock(in_c = 1, out_c = 8,dropout = 0)
        self.encoder2 = EncoderBlock(in_c = 8, out_c = 16, dropout = 0)
        self.tempencoding = nn.Conv3d(in_channels=16, out_channels=16 * 4, kernel_size=(4, 1, 1), stride=1)
        # Decoder
        self.dblock1=DecoderBlock(16*4,8,padding=(0,0,1))
        self.dblock2=DecoderBlock(8,1,padding=(1,1,1))

    def forward(self, x):
        # Spatial Encoding:
        x_t = torch.chunk(x, 4, dim=1)  # splits x into a tuple of 4 [1,1,61,81,31] tensors along the time dimension
        x_t = list(x_t) # convert x_t to a list so i can modify the elements


        for i in range(4):         # generates for each timestep a squeezed tensor of [1,61,81,31] and spatially encodes all tensors
            x_t[i] = torch.squeeze(x_t[i],1)
            x_t[i] = self.encoder1(x_t[i])
            #if i ==0:
                #print(f'after first encoding: {x_t[0].shape}')
            x_t[i] = self.encoder2(x_t[i])
            #if i ==0:
                #print(f'after second encoding: {x_t[0].shape}')
        x = torch.cat([tensor.unsqueeze(-5) for tensor in x_t], dim=-5) # concats along the time to [batch,time, channels ,x,y,z]
        #print(f'here {x.shape}')
        # Temporal Encoding:
        x = x.view(x.shape[0], 16, 4, 1, self.input_dim)  # reshape the dimensions such that a 3D layer can process the time
        x = self.tempencoding(x) # reduces the time dimension to 1
        x = x.view(x.shape[0], -1, self.d1, self.d2, self.d3) # reshapes the tensor to [batch, output_channels, x,y,z]
        # Decoder:
        x = self.dblock1(x)
        x = self.dblock2(x)

        return x

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
class UNet_timeconv_cross(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # cross means it has crossconnections between layers
    def __init__(self, d1 = 61, d2 = 81, d3 = 31):
        super(UNet_timeconv_cross, self).__init__()
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

class UNet_timeconv_deviation(BaseModel):
    # this is a new version of CNN3D1D with cross connections, but only 2 blocks deep
    # maxpooling instead of step size and additional conv layers in each block
    # work in progress
    def __init__(self, d1 = 61, d2 = 81, d3 = 31):
        super(UNet_timeconv_deviation, self).__init__()
        # initialize the latent space dimensions:
        self.d1 = math.floor(math.floor(d1/2)/2)
        self.d2 = math.floor(math.floor(d2/2)/2)
        self.d3 = math.floor(math.floor(d3/2)/2)
        self.input_dim = self.d1*self.d2*self.d3
        # Encoder
        self.encoder1 = EncoderBlock(in_c = 1, out_c = 8,dropout = 0)
        self.encoder2 = EncoderBlock(in_c = 8, out_c = 16, dropout = 0)
        self.tempencoding = nn.Conv3d(in_channels=16, out_channels=16 * 4, kernel_size=(4, 1, 1), stride=1)
        # Decoder
        self.dblock1=DecoderBlock(16*4,8,padding=(0,0,1))
        self.dblock2=DecoderBlock(8,1,padding=(1,1,1))

    def forward(self, x):
        # Spatial Encoding:
        x_t = torch.chunk(x, 4, dim=1)  # splits x into a tuple of 4 [batch, 1,1,61,81,31] tensors along the time dimension
        x_4 = x_t[3].squeeze(dim =1)
        x_t = list(x_t) # convert x_t to a list so i can modify the elements


        for i in range(4):         # generates for each timestep a squeezed tensor of [1,61,81,31] and spatially encodes all tensors
            x_t[i] = torch.squeeze(x_t[i],1)
            x_t[i] = self.encoder1(x_t[i])
            #if i ==0:
                #print(f'after first encoding: {x_t[0].shape}')
            x_t[i] = self.encoder2(x_t[i])
            #if i ==0:
                #print(f'after second encoding: {x_t[0].shape}')
        x = torch.cat([tensor.unsqueeze(-5) for tensor in x_t], dim=-5) # concats along the time to [batch,time, channels ,x,y,z]
        #print(f'here {x.shape}')
        # Temporal Encoding:
        x = x.view(x.shape[0], 16, 4, 1, self.input_dim)  # reshape the dimensions such that a 3D layer can process the time
        x = self.tempencoding(x) # reduces the time dimension to 1
        x = x.view(x.shape[0], -1, self.d1, self.d2, self.d3) # reshapes the tensor to [batch, output_channels, x,y,z]
        # Decoder:
        x = self.dblock1(x)
        x = self.dblock2(x)

        return x+x_4
class CNN3D_encoder_4(nn.Module):
    # encoder that spatially encodes one tensor and gives all inbetween tensors as outputs
    def __init__(self):
        super(CNN3D_encoder_4,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 4, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding = 2, padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding = 0)
    def forward(self, x1):
        x2 = F.relu(self.conv1(x1))
        x3 = F.relu(self.conv2(x2))
        x4 = F.relu(self.conv3(x3))

        # xi with i refering to the timestep <- no
        return x1,x2,x3,x4
class CNN3D_decoder_4(nn.Module):
    def __init__(self):
        super(CNN3D_decoder_4, self).__init__()
        #self.factor = 3
        self.deconv1 = nn.ConvTranspose3d(in_channels= 4 * 32, out_channels= 16,
                                          kernel_size=(2, 2, 2), stride=2,
                                          padding=0, output_padding=(1, 0, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=5 * 16, out_channels= 8, kernel_size=(2, 2, 2),
                                          stride=2,
                                          padding=2, output_padding=(0, 0, 1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=5 * 8, out_channels=1, kernel_size=3, stride=2,
                                          padding=4,
                                          output_padding=0)

        self.conv1dfake = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(5, 1, 1), stride=1)

    # xij with i is the timestep j is the deepness of the encoding layer
    def forward(self, x14,x24,x34,x44, x13,x23,x33,x43, x12,x22,x32,x42,  x11,x21,x31,x41):
        # concats all four timesteps with the output of the pre
        y3 = F.relu(self.deconv1(torch.cat([x14,x24,x34,x44], dim=1)))
        y2 = F.relu(self.deconv2(torch.cat([y3,x13,x23,x33,x43], dim=1)))
        y1 = F.relu(self.deconv3(torch.cat([y2, x12, x22, x32, x42], dim=1)))

        y = torch.cat([y1,x11,x21,x31,x41])

        #1D conv
        y= y.reshape(-1,1,5,1,61*81*31)
        y= self.conv1dfake(y)
        y= y.reshape(-1,1,61,81,31)     #reshape s.t. conv1 accepts the input

        return y
class CNN1D3D(nn.Module):
    def __init__(self):
        super(CNN1D3D,self).__init__()
        self.channels = 16
        self.conv1dfake =nn.Conv3d(in_channels= 1, out_channels= self.channels,kernel_size= (4,1,1),stride= 1) #padding= (4,0,0)) #kernel size resembles a 1D Conv over first dimension ---> NEW CHANNELS IN 3D!!!
        #self.p3d = (4,4,4,4,4,4) #3d padding on each side with length 8

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=self.channels*2*2, kernel_size=2, stride=2, padding = 8, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=self.channels*2*2, out_channels=self.channels*4*2, kernel_size=2, stride=2, padding = 4,padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=self.channels*4*2, out_channels=self.channels*8*2, kernel_size=2, stride=2, padding = 0)
        #here paper flattens the output and uses a fc layer
        self.deconv1= nn.ConvTranspose3d(in_channels=self.channels*8*2, out_channels=self.channels*4*2,kernel_size=2,stride=2, padding= 0, output_padding=(1,0,1))
        self.deconv2= nn.ConvTranspose3d(in_channels=self.channels*4*2, out_channels=self.channels*2*2, kernel_size=2, stride=2 , padding = 4, output_padding=(0,0,1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=self.channels*2*2, out_channels=1, kernel_size=3, stride=2, padding = 8, output_padding=0)

        #self.conv_end =nn.Conv3d(in_channels=5, out_channels =1, kernel_size=1,stride=1)
        #self.conv1dfake2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 1), stride=1)


    def forward(self,x1):
        # Input =       (batchsize, 4, 1, 61, 81, 31)   =   (batch, timesteps, channel, x, y, z)
        x= x1.reshape(-1,1,4,1,61*81*31)
        x= self.conv1dfake(x)
        x= x.reshape(-1,self.channels,61,81,31)     #reshape s.t. conv1 accepts the input
        #x = F.pad(x, self.p3d, "constant", 0) #padding with constant 30
        x = self.conv1(x)
        x = F.relu(x)
        #x = F.pad(x, self.p3d, "constant", 0) #padding with constant 30
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)

        #x1 = x1.reshape(-1,4,61,81,31)
        #x = torch.cat([x1,x], dim = 1)
        #x = x.reshape((-1,1,5,61*81*31))
        #x = F.relu(self.conv1dfake2(x))
        #x = x.reshape(-1,1,61,81,31)
        return x

    def __str__(self):
        return "1d convolution for the time dimension, 3 convolution, 3 deconvolution CNN"
# padding 4/3/3
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 4, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding = 2, padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding = 0,padding_mode='replicate')
        #here paper flattens the output and uses a fc layer

        self.deconv1= nn.ConvTranspose3d(in_channels=32, out_channels=16,kernel_size=(2,2,2),stride=2 , padding = 0, output_padding=(1,0,1))
        self.deconv2= nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 2, output_padding=(0,0,1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(3,3,3), stride=2, padding = 4)


    def forward(self,x):
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(x)
        x = self.conv2(x)
        #print(x.size())
        x = F.relu(x)
        x = self.conv3(x)
        #print(x.size())
        x = F.relu(x)
        x = self.deconv1(x)
        #print(x.size())
        x = F.relu(x)
        x = self.deconv2(x)
        #print(x.size())
        x = F.relu(x)
        x = self.deconv3(x)
        #print(x.size())
        x = F.relu(x)

        return x
    def __str__(self):
        return "3 convolution, 3 deconvolution CNN"
# no padding

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
class UNetLSTM(nn.Module):
    def __init__(self, h1 = 61,h2 =81,h3 = 31, out_c=4):
        super(UNetLSTM, self).__init__()
        self.h1 = math.floor(math.floor(h1/2)/2)
        self.h2 = math.floor(math.floor(h2/2)/2)
        self.h3 = math.floor(math.floor(h3/2)/2)
        self.input_dim = self.h1*self.h2*self.h3

        self.out_c = out_c  # number of final output channels in the encoder (relevant for the number of RNN cells)

        self.block1=EncoderBlock(1,8)
        self.block2=EncoderBlock(8,out_c)

        # a list with out_c lstms one for each output_channel from the encoder (doesn't work because of memory overload)
        self.lstms = nn.ModuleList([nn.LSTM(self.input_dim, self.input_dim, batch_first=True) for _ in range(out_c)])
        # a fully connected list for each output_channel to generate the prediction of the lstm
        self.fcs = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(out_c)])

        self.dblock1=DecoderBlock(out_c,8,padding=(0,0,1))
        self.dblock2=DecoderBlock(8,1,padding=(1,1,1))


        #self.decoder_up1 = nn.ConvTranspose3d(rnn_hidden_size, 64, kernel_size=2, stride=2)
        #self.output_conv = nn.Conv3d(64, out_channels, kernel_size=1)


    def forward(self, x):
        x_t = torch.chunk(x, 4, dim=1)  # splits x into a tuple of 4 [1,1,61,81,31] tensors along the time dimension
        x_t = list(x_t) # convert x_t to a list so i can modify the elements


        for i in range(4):         # generates for each timestep a squeezed tensor of [1,61,81,31] and spatially encodes all tensors
            x_t[i] = torch.squeeze(x_t[i],1)
            x_t[i] = self.block1(x_t[i])
            if i ==0:
                print(f'after first encoding: {x_t[0].shape}')
            x_t[i] = self.block2(x_t[i])
            if i ==0:
                print(f'after second encoding: {x_t[0].shape}')

        x = torch.cat([tensor.unsqueeze(-5) for tensor in x_t], dim=-5) # concats along the time to [batch,time, channels ,x,y,z]
        print(f'here {x.shape}')
        # flattens x such that it can be processed by the lstm
        x = x.view(x.shape[0],x.shape[1],-1) # dimensions: [batch, time, channels*x*y*z]
        # here i want a list encoder_output_list =[] with out_c elements of size [batch, time, x*y*z]
        encoder_output_list = list(torch.split(x, split_size_or_sections=self.input_dim, dim=-1))


        # apply as much different lstms as we have output channels in the encoder
        lstm_outputs = []
        for i in range(self.out_c):
            lstm_output, _ = self.lstms[i](encoder_output_list[i])
            lstm_output = self.fcs[i](lstm_output[:, -1, :]) # take the last one, because that's our prediction for the future timestep
            lstm_outputs.append(lstm_output) # this is a list of out_c tensors with shape [batch,x*y*z] where x,y,z are the dimensions of the latent space

        x = torch.cat([tensor.unsqueeze(-2) for tensor in lstm_outputs], dim=-2) # concat all lstm outputs to form out_c channels to get [batch, channel,x*y*z*]
        # reconstruct the shape [batch, channels,x,y,z]
        x = x.view(x.shape[0],x.shape[1],self.h1,self.h2,self.h3)
        x = self.dblock1(x)
        print(f'after first decoder: {x.shape}')
        x = self.dblock2(x)
        print(f'after second decoder: {x.shape}')

        return x

if __name__ == '__main__':
    # work in progress on UNet_timeconv

    #model = UNetLSTM().cuda()
    model = UNet_timeconv_deviation().cuda()
    #summary(model, (4, 1, 61, 81, 31))
    x = torch.arange(1*4*61*81*31).reshape(1, 4, 1, 61, 81, 31).cuda() #(batch, timesteps,input_channels, 61,81,31)
    y = model(x.float())
    print(y.shape)

    #model2 = CNN1D3D().cuda()
    #summary(model, (4, 1, 61, 81, 31))


