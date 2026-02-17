from torch import nn
import torch

#Single block of the decoder model
#Composition: CONV2D -> RELU -> CONV2D -> RELU
def decoder_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )

#Class that create the forward pass of the decoder
class Modello_0_U_net_decoder(nn.Module):
    def __init__(self, input_shape:int, NUM_CLASS:int):
        super().__init__()

        #Decoder:

        #Up-Conv -> (1024 -> 512)
        self.unpool1 = nn.ConvTranspose2d(in_channels=input_shape, out_channels=512, kernel_size=2, stride=2)
        self.dec1 = decoder_block(1536, 512) # 512 from unpool + 1024 from skip

        self.unpool2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec2 = decoder_block(768, 256) # 256 from unpool + 512 from skip

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = decoder_block(384, 128) # 128 from unpool + 256 from skip

        self.unpool4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec4 = decoder_block(192, 64) # 64 from unpool + 128 from skip

        self.unpool5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(64, NUM_CLASS, kernel_size=1)

    #Forward path
    def forward(self, x10, x9, x7, x5, x3):     #x10, x9, x7, x5, x3 is used for the skip connections

        #Decoder livello 1 (1024 -> 512)
        x = self.unpool1(x10)
        x = torch.cat([x, x9], dim=1)
        x = self.dec1(x)

        #Decoder livello 2 (512 -> 256)
        x = self.unpool2(x)
        x = torch.cat([x, x7], dim=1)
        x = self.dec2(x)

        #Decoder livello 3 (256 -> 128)
        x = self.unpool3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dec3(x)

        #Decoder livello 4 (128 -> 64)
        x = self.unpool4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec4(x)

        x = self.unpool5(x)

        #Conv 1x1
        out = self.conv1(x)

        return out