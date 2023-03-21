import imageio
import os
import sys

images = []
filenames = [sys.argv[1] + '/' + i for i in os.listdir(sys.argv[1])]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(sys.argv[2], images)