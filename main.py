"""This script is for generating png heightmaps based on edge
detection as well as roberts segmentation, a file should be
placed in the root directory with the title of img.png"""
# All imports
import skimage
from skimage.future import graph
from skimage.segmentation import slic
from skimage import filters, color, img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = Image.open("img.png") #Open Image File
np_img = np.asarray(img.convert('RGB')) #Convert to RGB colour space for later processing
#segmentation stage
#SLIC image segmentation
labels1 = slic(np_img, compactness=30, n_segments=400, start_label=1) 
#colour reduction
g = graph.rag_mean_color(np_img, labels1) 
labels2 = graph.cut_threshold(labels1, g, 50)
out2 = color.label2rgb(labels2, np_img, kind='avg', bg_label=0) 
edges = filters.roberts(color.rgb2gray(out2)) #Roberts filtering for basic colour bordering to enhance tactile interpretation
edges_base = filters.roberts(color.rgb2gray(np_img)) #roberts filtering to reintroduce the textures of the image such as hair, paint strokes etc.
chull_diff = img_as_float(edges_base.copy()) 
chull_diff[edges.astype(bool)] = 0.5
skimage.io.imsave("img3.png", chull_diff)