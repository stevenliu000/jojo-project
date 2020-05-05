# %%
import os
import numpy as np

import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

# %%
transf = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256), torchvision.transforms.ToTensor(),])

# %%
test_dataset = torchvision.datasets.ImageFolder(root='./bbbb/', 
                                                 transform=transf)

# %%
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                               shuffle=False, num_workers=1)

# %%
vgg19 = torchvision.models.vgg19_bn(pretrained=True)
vgg19.linear = nn.Identity()

# %%
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg19 = vgg19.to(DEVICE)

# %%
# SIMPLE MODEL DEFINITION
class jojo_MLP(nn.Module):
    def __init__(self, size_list):
        super(jojo_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).double()

# %%
jojo = jojo_MLP([1000, 512, 256, 5])
jojo.load_state_dict(torch.load("./jojo_classify.pt"))
jojo = jojo.to(DEVICE)
jojo.eval()

# %%
def test(data_loader):
    vgg19.eval()

    for batch_num, (feats, labels) in tqdm(enumerate(data_loader)):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            z = vgg19(feats).detach()
            y = jojo(z).detach()
        pred = F.softmax(y, dim=1)[0].detach().cpu().numpy() 


        torch.cuda.empty_cache()
        del feats
        del labels
        print("score: ")
        print(pred)

# %%
test(test_dataloader)
