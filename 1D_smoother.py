import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F



class ConvAutoencoder(nn.Module):
    def __init__(self, output_channels=8, dim_reduction=2, kernel_size=4, cross = True): # kernel size open for testint (2, 8 or even changing kernel size in each encoding block from to 3 or 2 or even 8, outputchannels can be changed to 16,32, dim_reduction open for testing with 4 (i wouldnt do more), cross = False for no crossconnections
        super(ConvAutoencoder, self).__init__()
        self.cross=cross

        # Encoder layers
        self.enc1_conv = nn.Conv1d(1, output_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.enc1_bn = nn.BatchNorm1d(output_channels)
        self.enc1_pool = nn.AvgPool1d(kernel_size, stride=dim_reduction)

        self.enc2_conv = nn.Conv1d(output_channels, output_channels*2, kernel_size=kernel_size, stride=1, padding='same')
        self.enc2_bn = nn.BatchNorm1d(output_channels*2)
        self.enc2_pool = nn.AvgPool1d(kernel_size, stride=dim_reduction)

        self.enc3_conv = nn.Conv1d(output_channels*2, output_channels*4, kernel_size=kernel_size, stride=1, padding='same')
        self.enc3_bn = nn.BatchNorm1d(output_channels*4)
        self.enc3_pool = nn.AvgPool1d(kernel_size, stride=dim_reduction)


        # Decoder layers
        self.dec1_upconv = nn.ConvTranspose1d(output_channels*4, output_channels*2, kernel_size=kernel_size, stride=dim_reduction, padding='same')
        self.dec1_bn = nn.BatchNorm1d(output_channels*2)
        # After concatenation, the number of channels doubles

        if cross == True:
            self.dec1_conv = nn.Conv1d(output_channels*2*2, output_channels*2, kernel_size=kernel_size, stride=1, padding='same') # output_channels = earlier channels times 2 because we concat the crossconnections additionally to this layer
        elif cross ==False:
            self.dec1_conv = nn.Conv1d(output_channels*2, output_channels*2, kernel_size=kernel_size, stride=1, padding='same')

        self.dec2_upconv = nn.ConvTranspose1d(output_channels*2, output_channels, kernel_size=kernel_size, stride=dim_reduction, padding='same')
        self.dec2_bn = nn.BatchNorm1d(output_channels)
        if cross == True:
            self.dec2_conv = nn.Conv1d(output_channels*2, output_channels, kernel_size=kernel_size, stride=1, padding='same') # output_channels = earlier channels times 2 because we concat the crossconnections additionally to this layer
        elif cross ==False:
            self.dec2_conv = nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, stride=1,
                                       padding='same')
        self.dec3_upconv = nn.ConvTranspose1d(output_channels, 1, kernel_size=kernel_size, stride=dim_reduction, padding='same')
        self.dec3_bn = nn.BatchNorm1d(1)
        if cross == True:
            self.dec3_conv = nn.Conv1d(1*2, 1, kernel_size=kernel_size, stride=1, padding='same') # output_channels = earlier channels times 2 because we concat the crossconnections additionally to this layer
        elif cross ==False:
            self.dec3_conv = nn.Conv1d(1 , 1, kernel_size=kernel_size, stride=1, padding='same')



    def forward(self, x):
        # Encoder
        e1 = F.relu((self.enc1_bn(self.enc1_conv(x))))
        e1p = self.enc1_pool(e1)

        e2 = F.relu((self.enc2_bn(self.enc2_conv(e1p))))
        e2p = self.enc2_pool(e2)

        e3 = F.relu((self.enc3_bn(self.enc3_conv(e2p))))
        e3p = self.enc3_pool(e3)


        # Decoder
        d1 = F.relu((self.dec1_bn(self.dec1_upconv(e3p))))
        if self.cross == True:
            d1 = torch.cat((d1, e3), 1)

        d1 = F.relu((self.dec1_conv(d1)))

        d2 = F.relu((self.dec2_bn(self.dec2_upconv(d1))))
        if self.cross == True:
            d2 = torch.cat((d2, e2), 1)
        d2 = F.relu((self.dec2_conv(d2)))

        d3 = F.relu((self.dec3_bn(self.dec3_upconv(d2))))
        if self.cross == True:
            d3 = torch.cat((d3, e1), 1)
        d3 = F.relu((self.dec3_conv(d3)))

        return d3

if __name__ =='__main__':

    x = torch.rand(1, 1, 1024) # create random tensor with the right dimensions to put into the model
    # Model instantiation
    model = ConvAutoencoder(output_channels=8, dim_reduction=2, kernel_size=4)
    summary(model, input_size=(1, 1024))
    y = model(x)
    print(y.shape)

    # Optimizer and learning rate scheduler setup
    #initial_learning_rate = 0.005
    # = optim.Adam(model.parameters(), lr=initial_learning_rate)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    #print(model)