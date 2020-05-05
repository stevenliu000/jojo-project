import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

transf = torchvision.transform.Compose([torchvision.transforms.RandomCrop(256), torchvision.transforms.ToTensor(),])
train_dataset = torchvision.dataset.ImageFolder(root='./aaaa/',transform = transf)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

for _,(a,b) in enumerate(train_dataloader):
    print(a.shape)
