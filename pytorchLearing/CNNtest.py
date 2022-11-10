import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('../testdata',train=False,transform=torchvision.transforms.ToTensor(),download= True)
dataloader = DataLoader(dataset,batch_size=64)

class CheatCheck(nn.Module):
    def __init__(self) :
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)

        return x


step =0
cheatcheck =CheatCheck()
print(cheatcheck)
writer = SummaryWriter('../log')


for data in dataloader:
    img ,targets = data
    output = cheatcheck(img)


