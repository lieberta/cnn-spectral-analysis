import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset_x4_y1, Dataset_x1_y1




class CNN3D_encoder_4(nn.Module):
    def __init__(self):
        super(CNN3D_encoder_4,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 4, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding = 2, padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding = 0)

    def forward(self, x1):
        x2 = F.relu(self.conv1(x1))
        x3 = F.relu(self.conv2(x2))
        x4 = F.relu(self.conv3(x3))

        # xi with i refering to the timestep
        return x1,x2,x3,x4
class CNN3D_decoder_4(nn.Module):
    def __init__(self):
        super(CNN3D_decoder_4, self).__init__()
        #self.factor = 3
        self.deconv1 = nn.ConvTranspose3d(in_channels=5 * 32, out_channels= 16,
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
    def forward(self, x_bot,x14,x24,x34,x44, x13,x23,x33,x43, x12,x22,x32,x42,  x11,x21,x31,x41):
        print(f'x_bot = {x_bot.size()}, x_14 = {x14.size()}')
        #y3 = F.relu(self.deconv1(torch.cat([x_bot,x14,x24,x34,x44], dim=1)))


        #1D conv
        #y= y.reshape(-1,1,5,1,61*81*31)
        #y= self.conv1dfake(y)
        #y= y.reshape(-1,1,61,81,31)     #reshape s.t. conv1 accepts the input

        return x_bot

class CNN3D1D(nn.Module):
    def __init__(self):
        super(CNN3D1D, self).__init__()
        self.block1 = CNN3D_encoder_4()
        self.block2 = CNN3D_encoder_4()
        self.block3 = CNN3D_encoder_4()
        self.block4 = CNN3D_encoder_4()

        self.convfake1d = nn.Conv3d(in_channels=32, out_channels=32 * 4 , kernel_size=(4, 1, 1), stride=1)

        self.block_deconv = CNN3D_decoder_4()

    def forward(self, x):
        # xi, i indicating the timestep
        x1 = x[:, 0, :, :, :, :]  # dimension [batch, in_channel = 1, 61,81,31]
        x2 = x[:, 1, :, :, :, :]
        x3 = x[:, 2, :, :, :, :]
        x4 = x[:, 3, :, :, :, :]

        # each block gives out four different size tensors from the different encoder layers
        x11,x12,x13,x14 = self.block1(x1)
        x21,x22,x23,x24 = self.block2(x2)
        x31,x32,x33,x34 = self.block3(x3)
        x41,x42,x43,x44 = self.block4(x4)

        x_bot = torch.stack((x14, x24, x34, x44), dim=1, out=None)  # stack them bois together to one fat tensor
        x_bot = x_bot.reshape(-1, 32, 4, 1, 9 * 12 * 5) #dimensions of last cnn layer
        x_bot = self.convfake1d(x_bot)
        x_bot = x_bot.reshape(-1, 32 * 4 , 9, 12, 5)

        y = self.block_deconv(x_bot,x14,x24,x34,x44, x13,x23,x33,x43, x12,x22,x32,x42,  x11,x21,x31,x41)

        return y

class CNNLSTM3D(nn.Module):
    def __init__(self):
        super(CNN1D3D,self).__init__()
        self.channels = 8
        self.conv1dfake =nn.Conv3d(1,self.channels,(4,1,1),1) #padding= (4,0,0)) #kernel size resembles a 1D Conv over first dimension ---> NEW CHANNELS IN 3D!!!
        #self.p3d = (4,4,4,4,4,4) #3d padding on each side with length 8

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=self.channels*8, kernel_size=(4,4,4), stride=2, padding = 4) #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=self.channels*8, out_channels=self.channels*8*2, kernel_size=(3,3,3), stride=2, padding = 4)
        self.conv3 = nn.Conv3d(in_channels=self.channels*8*2, out_channels=self.channels*8*4, kernel_size=(3,3,3), stride=2, padding = 4)
        #here paper flattens the output and uses a fc layer
        self.deconv1= nn.ConvTranspose3d(in_channels=self.channels*8*4, out_channels=self.channels*8*2,kernel_size=(3,3,3),stride=2, padding= 4, output_padding=(1,0,1))
        self.deconv2= nn.ConvTranspose3d(in_channels=self.channels*8*2, out_channels=self.channels*8, kernel_size=(3,3,3), stride=2 , padding = 4, output_padding=(0,0,1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=self.channels*8, out_channels=1, kernel_size=(4,4,4), stride=2, padding = 4, output_padding=1)

    def forward(self,x):
        # Input =       (batchsize, 4, 1, 61, 81, 31)   =   (batch, timesteps, channel, x, y, z)
        x= x.reshape(-1,1,4,1,61*81*31)
        x= self.conv1dfake(x)
        x= x.reshape(-1,self.channels,61,81,31)     #reshape s.t. conv1 accepts the input


        x = F.relu(x)
        return x

    def __str__(self):
        return "1d convolution for the time dimension, 3 convolution, 3 deconvolution CNN"

class CNN3DLSTM(nn.Module):
    def __init__(self):
        super(CNN3DLSTM,self).__init__()

        self.block1 = CNN3D_encoder()
        self.block2 = CNN3D_encoder()
        self.block3 = CNN3D_encoder()
        self.block4 = CNN3D_encoder()
        #here paper flattens the output and uses a fc layer

        #here to each temperature of a gridpoint the locality should be added, i.e. (x,y,z,temp) as input
        self.LSTM = nn.LSTM(input_size=4, hidden_size= 4, num_layers=2, batch_first=True)
        #

        self.block_deconv = CNN3D_decoder(4)

    def forward(self,x):

        x1 = x[:,0,:,:,:,:] # dimension [batch, in_channel = 1, 61,81,31]
        x2 = x[:,1,:,:,:,:]
        x3 = x[:,2,:,:,:,:]
        x4 = x[:,3,:,:,:,:]

        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        x4 = self.block4(x4) # output torch.Size([batch= 1, output layer= 32, 10, 13, 6])

        # take the output of each block, put it into a sequence length of 4
        # (batch,10*13*6,sequlen=4,feature_dim=4)
        # one input tensor x = ((x1,y1,z1,t111_1),(x1,y1,z1,t111_2),(x1,y1,z1,t111_3),(x1,y1,z1,t111_4))
        '''für meinen erstes LSTM Paket: gehe über t'''

        '''
        x = torch.stack((x1,x2,x3,x4), dim=1,out= None) # stack them bois together to one fat tensor
        x = x.reshape(-1,32,4,1,10*13*6)
        x = self.convfake1d(x)
        x = x.reshape(-1,32*4,10,13,6)
        '''
        #x = self.block_deconv(x)

        return x1



class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 4, padding_mode='replicate') #kernel_size=4, stride=2
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding = 2, padding_mode='replicate')
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding = 0,padding_mode='replicate')
        #here paper flattens the output and uses a fc layer

        self.deconv1= nn.ConvTranspose3d(in_channels=32, out_channels=16,kernel_size=(2,2,2),stride=2 , padding = 0, output_padding=(1,0,1))
        self.deconv2= nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(2,2,2), stride=2, padding = 2, output_padding=(0,0,1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding = 4)


    def forward(self,x):
        x = self.conv1(x)
        print(x.size())
        x = F.relu(x)
        x = self.conv2(x)
        print(x.size())
        x = F.relu(x)
        x = self.conv3(x)
        print(x.size())
        x = F.relu(x)
        x = self.deconv1(x)
        print(x.size())
        x = F.relu(x)
        x = self.deconv2(x)
        print(x.size())
        x = F.relu(x)
        x = self.deconv3(x)
        print(x.size())
        x = F.relu(x)

        return x
    def __str__(self):
        return "3 convolution, 3 deconvolution CNN"




if __name__ == '__main__':

    model = CNN3D1D()
    model = model.double()

    dataset = Dataset_x4_y1()
    #dataset = Dataset_x1_y1()
    x,_ = dataset[0]

    x = x[None,:,:,:,:,:] # for dataset x4,y1
    print(f'input tensor={x.size()}')
    #x = x[None,:,:,:,:]
    y = model(x)
    print(f'output tensor={y.size()}')







    #x = x.reshape(1,4,1,61,81,31)

    #x = x[:,3,:,:,:,:]
    #print(x.size())
    #print(y.size())

    #print(torch.eq(x,y))

   # with torch.no_grad():

        #model = LSTM(seq_len)
       # x = torch.rand((1,4,seq_len))
        #y = model(x)
        #y = torch.tensor(y)
        #y = torch.tensor(y,requires_grad=False)
        #print(y[1][1].size())

