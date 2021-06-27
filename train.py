import torch
import torchvision
import argparse

from box import BoundingBox
from getbatch import getBatch
from wrapper import Wrapper

def parserArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="")
    parser.add_argument('-d', type=str, help="annotations path")
    parser.add_argument('-i', type=str, help="images path")
    parser.add_argument('-b', type=int, help="batch size")
    parser.add_argument('--pic', type=str, help="image path for test", default=None)
    parser.add_argument('--save', type=str, help="save path for .pth and png files")
    parser.add_argument('--test', action="store_true", help="tests recognition on entire dataset")
    parser.add_argument('--max-iter', type=int, help="maximum int of iterations")

    parser.add_argument('-c', action='store_true', help="loading weights from checkpoint")
    parser.add_argument('-p', type=str, help="path to checkpoint")
    parser.add_argument('-lr', type=float, help="learning rate")
    args = parser.parse_args()
    return args

class Trainer():
    def __init__(self, args, **kwargs):
        self.batchgen = getBatch(an_path=args.d, im_path=args.i)
        self.wrapper = Wrapper(checkpoint=args.c, checkpoint_path=args.p, lr=args.lr)
        self.box = BoundingBox()
        self.batch_size = args.b
        self.resize = torchvision.transforms.Resize((224,224))

    def createBox(self, path, coords, count=None, fname=None):
        self.box.createImage(path, coords, count, fname)

    def testPic(self, data):
        data = self.resize(data)
        print(data[None,...].shape)
        return self.wrapper.test(x=data[None,...].cuda())[0]

    def test(self):
        data = []
        target = []

        tensor = self.batchgen.getTest()
        path = tensor['image']
        data.append(self.resize(tensor['imageTensor'].cuda()))
        target.append(tensor['data'].cuda())

        self.path = path
        data = torch.stack(data)
        target = torch.stack(target)
        self.test_path = tensor['image']
        return self.wrapper.test(x=data, target=target)

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

args = parserArgs()
mod = Trainer(args)

if(not args.test and args.pic is None):
    i = 0
    while(i < 100000):
        hit, loss, out, target = mod.train()
        if( i % 1000 == 0 ):
            print("##################################################################################################################")
            print()
            print( "Iteration: " + str(i) )
            print()
            print("IoU: " + str(hit) + " %", "Loss: " + str(loss))
            print("Out: " + str(out.data))
            print("Target: " + str(target.data))
            print()
            print()
            mod.wrapper.save_stat(i, args.save)
            path = args.save + "/image" + str(i) + ".png"
            mod.createBox(path=mod.path, coords=[out, target], count=i)
        i += 1
elif(args.test and args.pic is not None):
    image = mod.batchgen.getPicByPath(args.pic)
    image.cuda()
    out = mod.testPic(image)
    mod.createBox(path=args.pic, coords=[out, out], fname="/storage/brno6/home/jakubsekula/test/testimage.png")
else:
    i = 0
    average = 0
    while(i < len(mod.batchgen.annotations.keys()) - 0):
        hit, out, target = mod.test()
        average += hit
        print("IoU: {:.2f}".format(hit) + " %", mod.test_path)
        i += 1
    print()
    print()
    average = float(average/len(mod.batchgen.annotations.keys()))
    print("Dataset average hit IoU: {:.2f}".format(average) + " %")
