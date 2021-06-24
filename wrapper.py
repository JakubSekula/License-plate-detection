import torch

from shapely.geometry import Polygon
from net import AlexNet as Alex

class Wrapper():
    def __init__(self, checkpoint=False, checkpoint_path="", lr=0.0001):
        self.net = Alex(checkpoint, checkpoint_path)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=lr)

    def area(self, a, b):
        #xmin, ymin, xmax, ymax
        polygon = Polygon([(a[0], a[1]), (a[2],a[1]), (a[2],a[3]), (a[0], a[3])])
        other_polygon = Polygon([(b[0], b[1]), (b[2],b[1]), (b[2],b[3]), (b[0], b[3])])
        intersection = polygon.intersection(other_polygon)
        union = polygon.union(other_polygon)
        return intersection.area / union.area * 100

    def train(self, x, target):

        self.optimizer.zero_grad()
        out = self.net.forward(x)
        lossfce = torch.nn.MSELoss()
        loss = lossfce(out, target)
        loss.backward()
        self.optimizer.step()
        
        return self.area(out[31], target[31]), loss, out[31], target[31]

    def save_stat(self, i):
        torch.save(self.net.net.state_dict(), "/storage/brno6/home/jakubsekula/test/checkpoint_" + str(i) + ".pth")