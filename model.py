# This file defines UNET Structure

import torch
import torch.nn as nn

"""Creating Convolutional Block"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size= 3, padding= 1) # this is 3 x 3 convolutional layer
        self.bn1 = nn.BatchNorm2d(out_c) # this is batch normalization layer

        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)  # this is 3 x 3 convolutional layer
        self.bn2 = nn.BatchNorm2d(out_c)  # this is batch normalization layer

        self.relu = nn.LeakyReLU()

    def forward(self, inputs): # this is a forward function where all the operations will take place; it takes the inputs which are previous feature maps
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) # its input will be the output of the first relu layer
        x = self.bn2(x)
        x = self.relu(x)
        #print(x.shape)

        return x

"""Creating Encoder Block"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

"""Creating Attention Block"""
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, out_c, in_c, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(out_c, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_c, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)
        self.sig= nn.Sigmoid()

    def forward(self, g, skip):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g = self.W_gate(g)
        x = self.W_x(skip)
        psi = self.relu(g + x)
        #psi = self.psi(psi)
        psi = self.sig(psi)
        psi = self.psi(psi)
        outputs = skip * psi
        return outputs

"""Creating Decoder Block"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, n_coefficients):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size= 2, stride= 2, padding= 0)# as we need to upsaple the feature map; therefore, we would be using transpose convolution
        self.att = AttentionBlock(out_c, in_c, n_coefficients)
        self.conv = conv_block(out_c + out_c, out_c) #input features is equal to 2 time output features


    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)

        return  x





"""Start Building U-Net Architecture"""
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder Part in Unet"""
        self.e1 = encoder_block(3, 64) # input channels = 3, and out put channels = 64
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.e5 = encoder_block(512, 1024)

        """Bottleneck"""
        #self.b = conv_block(512, 1024)
        self.b = conv_block(1024, 2048)
        """Decoder"""
        self.d1 = decoder_block(2048, 1024, 512)
        self.d2 = decoder_block(1024, 512, 256)
        self.d3 = decoder_block(512, 256, 128)
        self.d4 = decoder_block(256, 128, 64)
        self.d5 = decoder_block(128, 64, 32)

        """Classifier"""
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs) #s1 works as skip connection
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)

        """Bottleneck"""
        b = self.b(p5)
        #print(s1.shape, s2.shape, s3.shape, s4.shape)
        #print(b.shape)

        """Decoder"""
        d1 = self.d1(b, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)

        outputs = self.outputs(d5)
        #print(d4.shape)
        return outputs



if __name__ == "__main__":
    #x is a feature map, x is input
    #x = torch.randn((2, 32, 128, 128)) # 2 is batch_size, number of channels is 32, size is 128 by 128
    x = torch.randn((2, 3, 512, 512))
    # we learn this function by backpropagation in deeplearning
    #f = encoder_block(32, 64) # f is function for convolution block, number of input channels is 32, number of output channels is 64
    f = build_unet()
    #y, p = f(x) # x is input to f; we have an input x, which predicts the output y with the help of a function f
    y = f(x)
    #print(y.shape, p.shape)
    print(y.shape)
