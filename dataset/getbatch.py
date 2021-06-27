from os import listdir
from os import getcwd

import matplotlib.pyplot as plt 
import cv2
import torch
import numpy as np
import random
from pathlib import Path
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
import xml.etree.ElementTree as ET

class getBatch():    
    def __init__(self, an_path, im_path):
        self.annotation_path = an_path
        self.iterator = 0
        self.images_path = im_path
        self.files = listdir(self.annotation_path)
        print("found " + str(len(self.files)) + " files")
        print("Getting coords ...")
        self.annotations = {}
        for grfile in self.files:
            tree = ET.parse(self.annotation_path + grfile)
            root = tree.getroot()
            array = []
            array.append(int(root[4][5][0].text.strip()))
            array.append(int(root[4][5][1].text.strip()))
            array.append(int(root[4][5][2].text.strip()))
            array.append(int(root[4][5][3].text.strip()))
            #[xmin, ymin, xmax, ymax]
            inner = {}
            inner['data'] = torch.tensor(array).to(torch.float32)
            img_path = self.annotation_path + grfile.split('.')[0] + ".png"
            inner['image'] = img_path.replace("annotations", "images")
            self.annotations[grfile.split(".")[0][4:]] = inner
        self.prepareMatrix()

    def prepareMatrix(self):
        for im in self.annotations.keys():
            image = cv2.imread(self.images_path+"Cars"+im+".png")
            transformer = transforms.ToTensor()
            image = transformer(image)
            self.annotations[im]['imageTensor'] = image
        return image

    def getPicByPath(self, path):
        image = cv2.imread(path)
        transformer = transforms.ToTensor()
        image = transformer(image)
        return image

    def getTest(self):
        index = self.iterator
        self.iterator += 1
        return self.annotations[str(index)]

    def getNext(self):
        index = random.randint(0, len(self.annotations.keys()) - 10)
        return self.annotations[str(index)] 
