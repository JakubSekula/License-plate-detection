import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = Image.open('/storage/brno6/home/jakubsekula/License-plate-detection/dataset/images/Cars0.png')

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((0,  0), 500, 266, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.savefig('test.png')