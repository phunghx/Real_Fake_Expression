import cv2
import os
import sys
from string import Template
from mosse import *
# first argument is the haarcascades path
from facedetector import *
import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

import glob
videofiles = glob.glob('../data/test/*.mp4')
#videofiles = ['../data/test/8841051988_HAPPINESS.mp4']
errorfiles = []

rect = (0,0,1,1)
rectangle = False
rect_over = False  
def onmouse(event,x,y,flags,params):
    global sceneImg,rectangle,rect,ix,iy,rect_over, roi,scene

    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
#            cv2.rectangle(sceneCopy,(ix,iy),(x,y),(0,255,0),1)
#            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
	
        

        	sceneCopy = sceneImg.copy()
        	cv2.rectangle(sceneCopy,(ix,iy),(x,y),(0,0,255),2)

        	rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))       
        	roi = sceneImg[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        	cv2.imshow('mouse input', sceneCopy)

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True

        sceneCopy = sceneImg.copy()
        cv2.rectangle(sceneCopy,(ix,iy),(x,y),(0,0,255),3)

        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))       
        roi = sceneImg[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        cv2.imshow('mouse input', sceneCopy)
        #cv2.imwrite('roi.jpg', roi)
    elif event == cv2.EVENT_RBUTTONDOWN:
	scene = False
	cv2.destroyWindow('mouse input')

minsize = 20
caffe_model_path = "./model"

threshold = [0.6, 0.7, 0.7]
factor = 0.709
    
caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)




for videofile in videofiles:
	#for infname in sys.argv[2:]:
	video_capture = cv2.VideoCapture(videofile)
        filesave = videofile.split('/')[-1][:-4]
        folderdata = '/data5/Real_Fake_expression/data/testImage/' + filesave
	make_sure_path_exists(folderdata)

        metadata = open('../data/testmetadata/' + filesave + '.txt','w')
	first = True
	count = 1
	while True:
	    # Capture frame-by-frame
	   ret, image = video_capture.read()
	   if image is None:
		break
	   img = np.copy(image)
	   if first:
		   sceneImg = np.copy(image)
		   img_matlab = img.copy()
        	   tmp = img_matlab[:,:,2].copy()
                   img_matlab[:,:,2] = img_matlab[:,:,0]
	           img_matlab[:,:,0] = tmp

		   boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)

		   first = False
		   if len(boundingboxes) == 0:
			cv2.namedWindow('mouse input')
			cv2.setMouseCallback('mouse input',onmouse)
			for( x, y, w, h ) in faces:
				cv2.rectangle(sceneImg, (x, y), (x + w, y + h), (255, 255, 0), 1)
				
			#cv2.namedWindow('video')
			cv2.imshow('mouse input', sceneImg)
			scene = True
			while scene:
				keyPressed = cv2.waitKey(5)
				#if keyPressed == ord('r'):
				#        scene = False
				#        cv2.destroyWindow('mouse input')
			faces = [rect]	
		   elif	len(boundingboxes) >1:
			choosebox = 0
			x,y,w,h = boundingboxes[0][0],boundingboxes[0][1],boundingboxes[0][2]-boundingboxes[0][0],boundingboxes[0][3]-boundingboxes[0][1] 
			boxsize = w*h
			for ix in range(1,len(boundingboxes)):
				x,y,w,h = boundingboxes[ix][0],boundingboxes[ix][1],boundingboxes[ix][2]-boundingboxes[ix][0],boundingboxes[ix][3]-boundingboxes[ix][1] 
				if (w*h) > boxsize:
					choosebox = ix
					boxsize = w*h
			faces = [[boundingboxes[choosebox][0],boundingboxes[choosebox][1],boundingboxes[choosebox][2]-boundingboxes[choosebox][0],boundingboxes[choosebox][3]-boundingboxes[choosebox][1]]]
		   else:
			faces = [[boundingboxes[0][0],boundingboxes[0][1],boundingboxes[0][2]-boundingboxes[0][0],boundingboxes[0][3]-boundingboxes[0][1]]]
		   ( x, y, w, h ) = faces[0]
		   x,y,w,h = int(x),int(y),int(w),int(h)
	     	   #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
		   x1,y1,x2,y2 = x, y, x + w, y + h
		   tracker = MOSSE(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),[x,y,x+w,y+h],'no name')
	    	   #outfname = "tmp/%s.faces.jpg" % os.path.basename(infname)
	  	   # Display the resulting frame
	   else :
	   	tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
	   	(x1,y1,x2,y2) = tracker.draw_state(image)
	   #cv2.imshow('Video', image)
	   cv2.imwrite(folderdata + '/%04d.jpg' % count, img)
	   outstr = (folderdata + '/%04d.jpg' % count) + " " + str(x1) + " " + str(y1) + " " + str(x2-x1) + " " + str(y2-y1) + "\n"
	   metadata.write(outstr)
	   count = count + 1
	   #if cv2.waitKey(1) & 0xFF == ord('q'):
	   #	break
	metadata.close()   
	video_capture.release()
	#cv2.destroyAllWindows()
	#exit()

#np.save('errors.npy',np.array(errorfiles))
#cv2.imwrite(os.path.expanduser(outfname), image)
