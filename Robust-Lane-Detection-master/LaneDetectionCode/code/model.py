import torch

from torch import nn                   ##Import pytorch Neural Network Module
import torch.nn.functional as F         ##Import activation and loss functions

# If you’re using negative log likelihood loss and log softmax activation, 
# then Pytorch provides a single function F.cross_entropy that combines the two.

# nn.Module (uppercase M) is a PyTorch specific concept, and is a class we’ll be using a lot.
# nn.Module is not to be confused with the Python concept of a (lowercase m) module, 
# which is a file of Python code that can be imported.

class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward():
        pass


class ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward():
        pass


class STRNN(nn.Module):                             ##Segnet Based nn
    def __init__(self, inChannels=3, outChannels=2):
        # call the parent constructor
        super(STRNN, self).__init__()


        ######## Encoder Architecture ##########


        # initialize first set of CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_1_1 = nn.Conv2d(in_channels=inChannels, out_Channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_1_1 = nn.ReLU()
        self.en_conv_1_2 = nn.Conv2d(in_channels=64, out_Channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_1_2 = nn.ReLU()
        self.en_maxpool_1 = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2)

        # initialize the SCNN layer
        self.scnn = SCNN()          ##incomplete

        # initialize second set of CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_2_1 = nn.Conv2d(in_channels=64, out_Channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_2_1 = nn.ReLU()
        self.en_conv_2_2 = nn.Conv2d(in_channels=128, out_Channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_2_2 = nn.ReLU()
        self.en_maxpool_2 = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2)

        # initialize third set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_3_1 = nn.Conv2d(in_channels=128, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_3_1 = nn.ReLU()
        self.en_conv_3_2 = nn.Conv2d(in_channels=256, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_3_2 = nn.ReLU()
        self.en_conv_3_3 = nn.Conv2d(in_channels=256, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_3_3 = nn.ReLU()
        self.en_maxpool_3 = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2)

        # initialize fourth set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_4_1 = nn.Conv2d(in_channels=256, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_4_1 = nn.ReLU()
        self.en_conv_4_2 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_4_2 = nn.ReLU()
        self.en_conv_4_3 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_4_3 = nn.ReLU()
        self.en_maxpool_4 = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2)

        # initialize fifth set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_5_1 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_5_1 = nn.ReLU()
        self.en_conv_5_2 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_5_2 = nn.ReLU()
        self.en_conv_5_3 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_relu_5_3 = nn.ReLU()
        self.en_maxpool_5 = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2)


        ####### Encoder Architecture Ends ##############

        ########## RNN Architecture ########

        # initialize the ConvLSTM module
        self.convLSTM = ConvLSTM()              #########incomplete

        ######### RNN Architecture Ends ###########
        
        ##################  Decoder Architecture  ###############

        # initialize first set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_maxunpool_1 = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.de_conv_5_1 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_5_1 = nn.ReLU()
        self.de_conv_5_2 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_5_2 = nn.ReLU()
        self.de_conv_5_3 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_5_3 = nn.ReLU()

        # initialize second set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_maxunpool_2 = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.de_conv_4_1 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_4_1 = nn.ReLU()
        self.de_conv_4_2 = nn.Conv2d(in_channels=512, out_Channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_4_2 = nn.ReLU()
        self.de_conv_4_3 = nn.Conv2d(in_channels=512, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_4_3 = nn.ReLU()

        # initialize third set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_maxunpool_3 = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.de_conv_3_1 = nn.Conv2d(in_channels=256, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_3_1 = nn.ReLU()
        self.de_conv_3_2 = nn.Conv2d(in_channels=256, out_Channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_3_2 = nn.ReLU()
        self.de_conv_3_3 = nn.Conv2d(in_channels=256, out_Channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_3_3 = nn.ReLU()

        # initialize fourth set of UNPOOL => CONV => RELU => CONV => RELU layers
        self.de_maxunpool_4 = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.de_conv_2_2 = nn.Conv2d(in_channels=128, out_Channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_2_2 = nn.ReLU()
        self.de_conv_2_3 = nn.Conv2d(in_channels=128, out_Channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_2_3 = nn.ReLU()

        # initialize fourth set of UNPOOL => CONV => RELU => CONV => RELU layers
        self.de_maxunpool_5 = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.de_conv_1_2 = nn.Conv2d(in_channels=64, out_Channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_relu_1_2 = nn.ReLU()
        self.de_conv_1_3 = nn.Conv2d(in_channels=64, out_Channels=2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.log_softmax = nn.LogSoftmax()

    ########## Decoder Architecture Ends ##########

    def forward(self, input):
        pass
