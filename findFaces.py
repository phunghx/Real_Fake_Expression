# import the necessary packages
from skimage import feature
import numpy as np
import cv2 
import os
folder_files = {'anger':'ANGER','contentment':'CONTENTMENT','disgust':'DISGUST','happy':'HAPPINESS','sadness':'SADNESS','surprise':'SURPRISE'}
import pickle
#import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_folder", required=True,
	help="path to image folder")
ap.add_argument("-s", "--save",  required=True,
	help="save pair of images to file")
args = vars(ap.parse_args())

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist

desc = LocalBinaryPatterns(24, 8)
#pathImage = '/server/dataset/Real_Fake/data/testImage/'
pathImage = args["image_folder"] + '/'
subjects = os.listdir(pathImage)

pairFaces = {}
while (len(subjects) > 0):
	subject = subjects[0]
	facial = subject.split('_')[1]
	img = cv2.imread(pathImage + subject + '/0001.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#hist = desc.describe(gray)
	dist_min = 0
	index = 0
	i = 0
	for i in range(len(subjects)):
		s = subjects[i]
		part = s.split('_')
	
		if s != subject and part[1]==facial:
			img2 = cv2.imread(pathImage + s + '/0001.jpg')
			gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
			gray2 = cv2.resize(gray2,(img.shape[0],img.shape[1]))
			#hist2 = desc.describe(gray2)
			#d = dist(hist,hist2)
			d = ssim(gray,gray2)
			if d > dist_min:
				dist_min = d
				pair = s
				index = i
	
	subjects.pop(index)
	subjects.pop(0)
	'''	
	fig = plt.figure()
	fig.add_subplot(1,2,1)
	plt.imshow(img)
	fig.add_subplot(1,2,2)
	plt.imshow(cv2.imread(pathImage + pair + '/0001.jpg'))
	plt.show()	
	'''
	pairFaces[subject] = pair
	pairFaces[pair] = subject
pickle.dump(pairFaces,open(args["save"],'wb'))
	



