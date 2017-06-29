import cv2
import os
import sys
from string import Template
from mosse import *
#import random
#random.seed(2)

# first argument is the haarcascades path
#face_cascade_path = sys.argv[1]
#face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
import dlib
scale_factor = 1.1
min_neighbors = 3
min_size = (200, 200)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

import errno
from skimage import io
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
detector = dlib.get_frontal_face_detector()
import glob
errorfiles = []
win = dlib.image_window()
def saveShape(shape,filename,size):
	data = []
	for i in range(68):
		data.append(1.0*shape.part(i).x/size[0])
	for i in range(68):	
		data.append(1.0*shape.part(i).y/size[1])
	data = np.array(data)
	np.save(filename,data)


videofiles = os.listdir('../data/testImage_v1/')
for videofile in videofiles:
	imgfiles = glob.glob('../data/testImage_v1/' + videofile + '/*.jpg')
	folderdata = '../data/testshape_v1/' + videofile + '/'
	make_sure_path_exists(folderdata)
	
	for imgfile in imgfiles:
	        filesave = imgfile.split('/')[-1][:-4]
		image = io.imread(imgfile)
		win.clear_overlay()
		win.set_image(image)
	   	d = dlib.rectangle(0,0,image.shape[0],image.shape[1])
	   	shape = predictor(image, d)
		#if videofile == "0020166400_ANGER" and filesave=="0002":
		#	import pdb;pdb.set_trace()
	        win.add_overlay(shape)
		saveShape(shape,folderdata + filesave,[image.shape[0],image.shape[1]])
	   	#dlib.hit_enter_to_continue()
