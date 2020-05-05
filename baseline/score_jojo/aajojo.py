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
train_dataset = torchvision.datasets.ImageFolder(root='./aaaa/', 
                                                 transform=transf)

# %%
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, 
                                               shuffle=True, num_workers=4)

# %%

# %%
vgg19 = torchvision.models.vgg19_bn(pretrained=True)

# %%
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
jojo = jojo.to(DEVICE)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(jojo.parameters())
numEpochs = 1

# %%
def train(data_loader):
    vgg19.train()
    jojo.train

    for epoch in tqdm(range(numEpochs)):
        for batch_num, (feats, labels) in enumerate(tqdm(data_loader)):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                z = vgg19(feats).detach()
            y = jojo(z)
            pred = F.softmax(y, dim=1)[0].detach().cpu().numpy() 
            
            loss = criterion(y, labels.long())
            loss.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            del feats
            del labels
            print("score: ")
            print(pred)
        print("loss: " + str(loss.item()))
        del loss

# %%
train(train_dataloader)

# %%
torch.save(jojo.state_dict(),"./jojo_classify.pt")
torch.save(vgg19.state_dict(),"./vgg19_classify.pt")
