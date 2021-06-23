from torch import nn
import torchvision
import torch
from getbatch import getBatch

class AlexNet(nn.Module):
    def __init__(self, checkpoint, checkpoint_path):
        super(AlexNet, self).__init__()
        self.net = torchvision.models.alexnet(pretrained=False, num_classes=4)
        if(checkpoint):
            self.net.load_state_dict(torch.load(checkpoint_path))
            self.net.eval()
        self.net.cuda()
        self.net.eval()
    def forward(self, x, **kwargs):
        return self.net(x)