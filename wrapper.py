import torch

from shapely.geometry import Polygon
from net import AlexNet as Alex

class Wrapper():
    def __init__(self, checkpoint, checkpoint_path, lr):
        self.net = Alex(checkpoint, checkpoint_path)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=lr)

    def area(self, a, b):
        #xmin, ymin, xmax, ymax
        polygon = Polygon([(a[0], a[1]), (a[2],a[1]), (a[2],a[3]), (a[0], a[3])])
        other_polygon = Polygon([(b[0], b[1]), (b[2],b[1]), (b[2],b[3]), (b[0], b[3])])
        intersection = polygon.intersection(other_polygon)
        union = polygon.union(other_polygon)
        return intersection.area / union.area * 100

    #def test(self, x, target):
    #    with torch.no_grad():
    #        out = self.net.forward(x)
    #    return self.area(out[0], target[0]), out[0], target[0]
    
    def test(self, x):
        with torch.no_grad():
            out = self.net.forward(x)
        return out

    def train(self, x, target):

        self.optimizer.zero_grad()
        out = self.net.forward(x)
        lossfce = torch.nn.MSELoss()
        loss = lossfce(out, target)
        loss.backward()
        self.optimizer.step()
        
        return self.area(out[31], target[31]), loss, out[31], target[31]

    def save_stat(self, i, path):
        torch.save(self.net.net.state_dict(), path + "/checkpoint_" + str(i) + ".pth")