import torch.nn as nn

from .decoder_V0 import Modello_0_U_net_decoder
from .encoder_V0 import Modello_0_U_net_encoder

#Define the class that compose the Encoder-Decoder model
class U_Net(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Modello_0_U_net_encoder(input_shape=3)                               #Encoder
        self.decoder = Modello_0_U_net_decoder(input_shape=1024, NUM_CLASS=num_classes)     #Decoder

    #Forward path
    def forward(self, x):
        x1, x3, x5, x7, x9, x10 = self.encoder(x)   #Encoder forward
        out = self.decoder(x10, x9, x7, x5, x3)     #Decoder forward
        return out