import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class BoundingBox():
    def __init__(self):
        ...

    def countCoords(self, out, target):
        #xmin,ymin,xmax,ymax
        width = target[2] - target[0]
        height = target[3] - target[1]
        green = [target[0], target[1], width, height]
        
        width = out[2] - out[0]
        height = out[3] - out[1]
        red = [out[0], out[1], width, height]
        return green, red

    def createImage(self, path, coords, count):
        green, red = self.countCoords(coords[0], coords[1])
        im = Image.open(path)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(im)

        # Create a Rectangle patch
        rect = patches.Rectangle((green[0], green[1]), green[2], green[3], linewidth=1, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((red[0], red[1]), red[2], red[3], linewidth=1, edgecolor='green', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(rect2)

        plt.savefig(fname="/storage/brno6/home/jakubsekula/test/trn_" + str(count) + ".png")
        plt.close()