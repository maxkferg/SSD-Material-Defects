import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

IMAGE_DEPTH = 1
LABEL_FILE = "ground_truth.txt"
DATA_DIR = os.path.expanduser("~/Data/GDXray/Castings")
rootdir = Path(DATA_DIR)


def to_edges(box):
	x1 = int(box[1])
	x2 = int(box[2])
	y1 = int(box[3])
	y2 = int(box[4])

	y = int(box[2])
	w = int(box[3])
	h = int(box[4])
	return [(x1,y1), (x2, y2)]


class Image:

	def __init__(self,filename,boxes):
		self.pixels = cv.imread(filename)
		self.width = self.pixels.shape[1]
		self.height = self.pixels.shape[0]
		self.depth = IMAGE_DEPTH
		self.filename = filename
		self.boxes = boxes

	def draw(self):
		for row in range(self.boxes.shape[0]):
			edges = to_edges(self.boxes[row,:])
			cv.rectangle(self.pixels, edges[0], edges[1], (0,0,255),2)
		self.pixels = self.pixels[:,:,::-1]
		plt.imshow(self.pixels)
		plt.show()


def get_images():
	"""Iterator for image objects"""

	for folder in rootdir.glob('C*'):
		if not folder.is_dir():
			continue
		label_file = folder/LABEL_FILE

		if label_file.exists():
			labels = np.loadtxt(label_file)
		else:
			labels = np.zeros((1,1))

		for i,filename in enumerate(folder.glob('*.png')):
			index = i+1
			boxes = labels[labels[:,0] == index]
			image = Image(str(filename),boxes)
			yield image



			
