import torch

from shapely.geometry import Polygon
from net import AlexNet as Alex

class Wrapper():
    def __init__(self, checkpoint, checkpoint_path, lr=0.001):
        self.net = Alex(checkpoint, checkpoint_path)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=lr)

    def area(self, a, b):
        #xmin, ymin, xmax, ymax
        polygon = Polygon([(a[0][0], a[0][1]), (a[0][2],a[0][1]), (a[0][2],a[0][3]), (a[0][0], a[0][3])])
        other_polygon = Polygon([(b[0][0], b[0][1]), (b[0][2],b[0][1]), (b[0][2],b[0][3]), (b[0][0], b[0][3])])
        intersection = polygon.intersection(other_polygon)
        return intersection.area

    def train(self, x, target):
        target = torch.tensor(target).cuda()
        target = target.to(torch.float32)
        target = target[None,...]
        self.optimizer.zero_grad()
        out = self.net.forward(x[None,...])
        lossfce = torch.nn.MSELoss()
        loss = lossfce(out, target)
        loss.backward()
        self.optimizer.step()
        
        return self.area(out, target), loss, out, target

    def save_stat(self, i):
        torch.save(self.net.net.state_dict(), "/storage/brno6/home/jakubsekula/test/checkpoint_" + str(i) + ".pth")
