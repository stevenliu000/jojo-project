import torch, utils
import torch.nn as nn
import torch.nn.functional as F
from conv2d_mtl import Conv2dMtl

'''
--------------------------------------------------------------------------------
code from pre-train-weight github
https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch/blob/master/test.py
'''
class Transformer(nn.Module):
    def __init__(self, conv=nn.Conv2d):
        super(Transformer, self).__init__()
        #
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = conv(3, 64, 7)
        self.in01_1 = InstanceNormalization(64)
        # relu
        self.conv02_1 = conv(64, 128, 3, 2, 1)
        self.conv02_2 = conv(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        # relu
        self.conv03_1 = conv(128, 256, 3, 2, 1)
        self.conv03_2 = conv(256, 256, 3, 1, 1)   
        self.in03_1 = InstanceNormalization(256)    
        # relu

        ## res block 1
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = conv(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        # relu
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = conv(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)
        # + input

        ## res block 2
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = conv(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        # relu
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = conv(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)
        # + input

        ## res block 3
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = conv(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        # relu
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = conv(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)
        # + input

        ## res block 4
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = conv(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        # relu
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = conv(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)
        # + input

        ## res block 5
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = conv(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        # relu
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = conv(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)
        # + input

        ## res block 6
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = conv(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        # relu
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = conv(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)
        # + input

        ## res block 7
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = conv(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        # relu
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = conv(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)
        # + input

        ## res block 8
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = conv(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        # relu
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = conv(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)
        # + input

        ##------------------------------------##
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = conv(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128)
        # relu
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = conv(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)
        # relu
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = conv(64, 3, 7)
        # tanh

    def forward(self, x):
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))

        ##
        y = F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(y))) + t04

        y = F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(y))) + t05

        y = F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(y))) + t06

        y = F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(y))) + t07

        y = F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(y))) + t08

        y = F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(y))) + t09

        y = F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(y))) + t10

        y = F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))
        y = self.in11_2(self.conv11_2(self.refpad11_2(y))) + t11
        ##

        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = F.tanh(self.deconv03_1(self.refpad12_1(y)))

        return y
    
class TransformerEncoder(nn.Module):
    def __init__(self, conv=nn.Conv2d):
        super(TransformerEncoder, self).__init__()
        #
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = conv(3, 64, 7)
        self.in01_1 = InstanceNormalization(64)
        # relu
        self.conv02_1 = conv(64, 128, 3, 2, 1)
        self.conv02_2 = conv(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        # relu
        self.conv03_1 = conv(128, 256, 3, 2, 1)
        self.conv03_2 = conv(256, 256, 3, 1, 1)   
        self.in03_1 = InstanceNormalization(256)    
        # relu

        ## res block 1
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = conv(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        # relu
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = conv(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)
        # + input

        ## res block 2
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = conv(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        # relu
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = conv(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)
        # + input

        ## res block 3
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = conv(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        # relu
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = conv(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)
        # + input

        ## res block 4
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = conv(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        # relu
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = conv(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)
        # + input

    def forward(self, x):
        net = {}
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))
        net['t04'] = t04
        
        ##
        y = F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(y))) + t04
        net['t05'] = t05

        y = F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(y))) + t05
        net['t06'] = t06

        y = F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(y))) + t06
        net['t07'] = t07

        y = F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(y))) + t07
        net['t08'] = t08

        return t08, net

class TransformerDecoder(nn.Module):
    def __init__(self, conv=nn.Conv2d):
        super(TransformerDecoder, self).__init__()
        ## res block 5
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = conv(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        # relu
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = conv(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)
        # + input

        ## res block 6
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = conv(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        # relu
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = conv(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)
        # + input

        ## res block 7
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = conv(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        # relu
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = conv(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)
        # + input

        ## res block 8
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = conv(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        # relu
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = conv(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)
        # + input

        ##------------------------------------##
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = conv(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128)
        # relu
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = conv(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)
        # relu
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = conv(64, 3, 7)
        # tanh

    def forward(self, t08):
        net = {}
        y = F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(y))) + t08
        net['t09'] = t09

        y = F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(y))) + t09
        net['t10'] = t10
        
        y = F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(y))) + t10
        net['t11'] = t11
        
        y = F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))
        y = self.in11_2(self.conv11_2(self.refpad11_2(y))) + t11
        net['t12'] = y
        ##

        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = F.tanh(self.deconv03_1(self.refpad12_1(y)))

        return y, net

class TransformerExplode(nn.Module):
    def __init__(self, conv=nn.Conv2d):
        super(TransformerExplode, self).__init__()
        #
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = conv(3, 64, 7)
        self.in01_1 = InstanceNormalization(64)
        # relu
        self.conv02_1 = conv(64, 128, 3, 2, 1)
        self.conv02_2 = conv(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        # relu
        self.conv03_1 = conv(128, 256, 3, 2, 1)
        self.conv03_2 = conv(256, 256, 3, 1, 1)   
        self.in03_1 = InstanceNormalization(256)    
        # relu

        ## res block 1
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = conv(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        # relu
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = conv(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)
        # + input

        ## res block 2
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = conv(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        # relu
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = conv(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)
        # + input

        ## res block 3
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = conv(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        # relu
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = conv(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)
        # + input

        ## res block 4
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = conv(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        # relu
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = conv(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)
        # + input

        ## res block 5
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = conv(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        # relu
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = conv(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)
        # + input

        ## res block 6
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = conv(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        # relu
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = conv(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)
        # + input

        ## res block 7
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = conv(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        # relu
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = conv(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)
        # + input

        ## res block 8
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = conv(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        # relu
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = conv(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)
        # + input

        ##------------------------------------##
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = conv(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128)
        # relu
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = conv(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)
        # relu
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = conv(64, 3, 7)
        # tanh

    def forward(self, x):
        net = {}
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))
        net['t04'] = t04
        
        ##
        y = F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(y))) + t04
        net['t05'] = t05

        y = F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(y))) + t05
        net['t06'] = t06

        y = F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(y))) + t06
        net['t07'] = t07

        y = F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(y))) + t07
        net['t08'] = t08

        y = F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(y))) + t08
        net['t09'] = t09

        y = F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(y))) + t09
        net['t10'] = t10
        
        y = F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(y))) + t10
        net['t11'] = t11
        
        y = F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))
        y = self.in11_2(self.conv11_2(self.refpad11_2(y))) + t11
        net['t12'] = y
        ##

        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = F.tanh(self.deconv03_1(self.refpad12_1(y)))

        return y, net
        

class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def __call__(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


'''
original code
-------------------------------------------------------------------------------
'''
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x #Elementwise Sum
 

class generator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3), #k7n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), #k3n128s2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), #k3n256s1
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), #k3n256s1
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), #k3n128s1/2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), #k3n64s1/2
            nn.Conv2d(nf, nf, 3, 1, 1), #k3n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
