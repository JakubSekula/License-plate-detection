import torch
import torchvision

from box import BoundingBox
from getbatch import getBatch
from wrapper import Wrapper

class Trainer():
    def __init__(self):
        self.batchgen = getBatch()
        self.wrapper = Wrapper()
        self.box = BoundingBox()
        self.batch_size = 32
        self.resize = torchvision.transforms.Resize((224,224))

    def createBox(self, path, coords, count):
        self.box.createImage(path, coords, count)

    def train(self):
        data = []
        target = []
        for i in range(0, self.batch_size):
            tensor = self.batchgen.getNext()
            path = tensor['image']
            data.append(self.resize(tensor['imageTensor'].cuda()))
            target.append(tensor['data'].cuda())
        self.path = path
        data = torch.stack(data)
        target = torch.stack(target)
        return self.wrapper.train(x=data, target=target)

mod = Trainer()
i=0
while(i < 100000):
    err, loss, out, target = mod.train()
    if( i % 1000 == 0 ):
        print("##################################################################################################################")
        print()
        print( "Iteration: " + str(i) )
        print()
        print("Hit: " + str(err) + " %", "Loss: " + str(loss))
        print("Out: " + str(out.data))
        print("Target: " + str(target.data))
        print()
        print()
        print("##################################################################################################################")
        mod.wrapper.save_stat(i)
        path = "/storage/brno6/home/jakubsekula/test/image" + str(i) + ".png"
        mod.createBox(path=mod.path, coords=[out, target], count=i)
    i += 1
