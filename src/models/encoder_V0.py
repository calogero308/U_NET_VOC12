from torch import nn

#Single block of the encoder model
#Composition: CONV2D -> RELU -> CONV2D -> RELU
def encoder_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=False),
    )

#Class that create the forward pass of the encoder
class Modello_0_U_net_encoder(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()

        # Encoder: decresing size
        self.enc1 = encoder_block(input_shape, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = encoder_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = encoder_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = encoder_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = encoder_block(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    #Forward path
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)

        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        x5 = self.enc3(x4)
        x6 = self.pool3(x5)

        x7 = self.enc4(x6)
        x8 = self.pool4(x7)

        x9 = self.enc5(x8)
        x10 = self.pool5(x9)

        #Return the parameters to create skip connections
        return x1, x3, x5, x7, x9, x10
