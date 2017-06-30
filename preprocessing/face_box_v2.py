import cv2
import os
import sys
from string import Template
from mosse import *
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
scale_factor = 1.1
min_neighbors = 3
min_size = (200, 200)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

import glob



import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video_folder", required=True,
	help="path to video folder")
ap.add_argument("-s", "--save",  required=True,
	help="path of face images to save")
args = vars(ap.parse_args())


videofiles = glob.glob( args["video_folder"] + '/*.mp4')

rect = (0,0,1,1)
rectangle = False
rect_over = False  
def checkInside(region,xc,yc):
	x,y,w,h = region[0],region[1],region[2],region[3]
	x2 = x+ w
	y2 = y +h
	if xc > x and xc<x2 and yc>y and yc<y2:
		return True
	return False
def onmouse(event,x,y,flags,params):
    global sceneImg,rectangle,rect,ix,iy,rect_over, roi,scene,boundingboxes,chooseindex

    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y
	chooseindex = 0

	for ii in range(len(boundingboxes)):
		if checkInside(boundingboxes[ii],ix,iy):
			chooseindex = ii
			break
	xc,yc,w,h = boundingboxes[chooseindex]
	sceneCopy = sceneImg.copy()
        cv2.rectangle(sceneCopy,(xc,yc),(xc+w,yc+h),(0,0,255),3)
	for ii in range(len(boundingboxes)):
		if ii != chooseindex:
			xc,yc,w,h = boundingboxes[ii]
		        cv2.rectangle(sceneCopy,(xc,yc),(xc+w,yc+h),(255, 255, 0),3)

	cv2.imshow('mouse input', sceneCopy)

#    elif event == cv2.EVENT_MOUSEMOVE:
#        if rectangle == True:
#            cv2.rectangle(sceneCopy,(ix,iy),(x,y),(0,255,0),1)
#            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
	
        

#        	sceneCopy = sceneImg.copy()
#        	cv2.rectangle(sceneCopy,(ix,iy),(x,y),(0,0,255),2)

#        	rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))       
#        	roi = sceneImg[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

#        	cv2.imshow('mouse input', sceneCopy)

   
    elif event == cv2.EVENT_RBUTTONDOWN:
	scene = False
	cv2.destroyWindow('mouse input')


for videofile in videofiles:
	#for infname in sys.argv[2:]:
	video_capture = cv2.VideoCapture(videofile)
        filesave = videofile.split('/')[-1][:-4]
        folderdata = args["save"] +'/' + filesave
	make_sure_path_exists(folderdata)

	first = True
	count = 1
	while True:
	    # Capture frame-by-frame
	   ret, image = video_capture.read()
	   if image is None:
		break
	   img = np.copy(image)
	   if first:
		   boundingboxes = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
		   	minSize = min_size, flags = flags)

		   first = False
		   if len(boundingboxes) <=0:
			sceneImg = np.copy(img)
			cv2.namedWindow('mouse input')
			cv2.setMouseCallback('mouse input',onmouse)
			for( x, y, w, h ) in boundingboxes:
				cv2.rectangle(sceneImg, (x, y), (x + w, y + h), (255, 255, 0), 1)
				
			#cv2.namedWindow('video')
			cv2.imshow('mouse input', sceneImg)
			scene = True
			while scene:
				keyPressed = cv2.waitKey(5)
				#if keyPressed == ord('r'):
				#        scene = False
				#        cv2.destroyWindow('mouse input')
			faces = [boundingboxes[chooseindex]]	
		   
		   elif len(boundingboxes)==1:
			faces = [[boundingboxes[0][0],boundingboxes[0][1],boundingboxes[0][2],boundingboxes[0][3]]]
		   else:
			selectIndex = 0
			yS = boundingboxes[0][1]
			for jj in range(1,len(boundingboxes)):
				if yS < boundingboxes[jj][1]:
					yS = boundingboxes[jj][1]
					selectIndex = jj

			faces = [[boundingboxes[selectIndex][0],boundingboxes[selectIndex][1],boundingboxes[selectIndex][2],boundingboxes[selectIndex][3]]]


		   (x, y, w, h ) = faces[0]
		   #x,y,w,h = int(x),int(y),int(w),int(h)
	     	   cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
		   x1,y1,x2,y2 = x, y, x + w, y + h
		   tracker = MOSSE(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),[x,y,x+w,y+h],'no name')
	    	   #outfname = "tmp/%s.faces.jpg" % os.path.basename(infname)
	  	   # Display the resulting frame
	   else :
	   	tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
	   	(x1,y1,x2,y2) = tracker.draw_state(image)
	   #cv2.imshow('Video', image)
	   cv2.imwrite(folderdata + '/%04d.jpg' % count, img[y1:y2,x1:x2,:])
	   #outstr = (folderdata + '/%04d.jpg' % count) + " " + str(x1) + " " + str(y1) + " " + str(x2-x1) + " " + str(y2-y1) + "\n"
	   #metadata.write(outstr)
	   count = count + 1
	   #if cv2.waitKey(1) & 0xFF == ord('q'):
	   #	break
	#metadata.close()   
	video_capture.release()
	#cv2.destroyAllWindows()
	#exit()

#np.save('errors.npy',np.array(errorfiles))
#cv2.imwrite(os.path.expanduser(outfname), image)
