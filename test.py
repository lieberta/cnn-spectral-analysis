
# Smoothing out time-series (frequencies of spectral analysis)
class ConvAutoencoder(Model):
    def __init__(self, output_channels=8, dim_reduction=2, kernel_size=4, cross=True):
        super(ConvAutoencoder, self).__init__()
        self.cross = cross

        # Encoder
        self.enc1_conv = Conv1D(output_channels, kernel_size, strides=1, padding='same')
        self.enc1_bn = BatchNormalization()
        self.enc1_pool = AveragePooling1D(pool_size=kernel_size, strides=dim_reduction, padding='same')

        self.enc2_conv = Conv1D(output_channels*2, kernel_size, strides=1, padding='same')
        self.enc2_bn = BatchNormalization()
        self.enc2_pool = AveragePooling1D(pool_size=kernel_size, strides=dim_reduction, padding='same')

        self.enc3_conv = Conv1D(output_channels*4, kernel_size, strides=1, padding='same')
        self.enc3_bn = BatchNormalization()
        self.enc3_pool = AveragePooling1D(pool_size=kernel_size, strides=dim_reduction, padding='same')

        # Decoder
        self.dec1_upconv = Conv1DTranspose(output_channels*2, kernel_size, strides=dim_reduction, padding='same')
        self.dec1_bn = BatchNormalization()
        # Adjust input channels based on cross-connection
        self.dec1_conv_channels = output_channels*4 if cross else output_channels*2
        self.dec1_conv = Conv1D(self.dec1_conv_channels, kernel_size, strides=1, padding='same')

        self.dec2_upconv = Conv1DTranspose(output_channels, kernel_size, strides=dim_reduction, padding='same')
        self.dec2_bn = BatchNormalization()
        self.dec2_conv_channels = output_channels*2 if cross else output_channels
        self.dec2_conv = Conv1D(self.dec2_conv_channels, kernel_size, strides=1, padding='same')

        self.dec3_upconv = Conv1DTranspose(1, kernel_size, strides=dim_reduction, padding='same')
        self.dec3_bn = BatchNormalization()
        self.dec3_conv_channels = 2 if cross else 1
        self.dec3_conv = Conv1D(self.dec3_conv_channels, kernel_size, strides=1, padding='same')

    def call(self, inputs):
        # inputs.dim =[1024,1]

        # Encoder
        x = self.enc1_conv(inputs)
        x = self.enc1_bn(x)
        x = ReLU()(x)
        e1 = self.enc1_pool(x)

        # e1.dim=[512,1] =

        x = self.enc2_conv(e1)
        x = self.enc2_bn(x)
        x = ReLU()(x)
        e2 = self.enc2_pool(x)

        # e2.dim=[256,1]

        x = self.enc3_conv(e2)
        x = self.enc3_bn(x)
        x = ReLU()(x)
        x = self.enc3_pool(x)

        #x.dim = [128,1]

        # Decoder
        x = self.dec1_upconv(x)

        if self.cross:
            x = Concatenate(axis=-1)([x, e2])
        x = self.dec1_conv(x)
        x = self.dec1_bn(x)
        x = ReLU()(x)


        x = self.dec2_upconv(x)
        if self.cross:
            x = Concatenate(axis=-1)([x, e1])
        x = self.dec2_conv(x)
        x = self.dec2_bn(x)
        x = ReLU()(x)

        x = self.dec3_upconv(x)
        if self.cross:
            x = Concatenate(axis=-1)([x,inputs])
        x = self.dec3_conv(x)
        x = self.dec3_bn(x)
        x = ReLU()(x)

        return x