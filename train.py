import torch

from box import BoundingBox
from getbatch import getBatch
from wrapper import Wrapper

class Trainer():
    def __init__(self):
        self.batchgen = getBatch()
        self.wrapper = Wrapper(checkpoint=True, checkpoint_path="/storage/brno6/home/jakubsekula/test/checkpoint_1000.pth")
        self.box = BoundingBox()

    def createBox(self, path, coords, count):
        self.box.createImage(path, coords, count)

    def train(self):
        self.current = self.batchgen.getNext()
        return self.wrapper.train(x=self.current['imageTensor'].cuda(), target=self.current['data'])

mod = Trainer()
i=0
while(i < 200000):
    err, loss, out, target = mod.train()
    if( i % 1000 == 0 ):
        print("Err: " + str(100 - err * 100) + " %", "Loss: " + str(loss))
        print( "Out: " + str(out[0].data))
        print( "Target: " + str(target[0].data) )
        print()
        print()
        mod.wrapper.save_stat(i)
        path = "/storage/brno6/home/jakubsekula/test/image" + str(i) + ".png"
        mod.createBox(path=mod.current['image'], coords=[out[0], target[0]], count=i)
    i += 1
