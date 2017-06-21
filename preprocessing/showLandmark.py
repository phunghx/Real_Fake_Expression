import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

faceRegionsFiles = glob.glob("../data/testmetadata/*.txt")
landMarkFiles = glob.glob("../data/testshape/*.txt")
folderSave = "../data/testshapedata/"



for i in range(len(faceRegionsFiles)):
	print(faceRegionsFiles[i])
	faceFile = faceRegionsFiles[i].split()[0]
	f = open(faceFile,'r')
	faceRegions = f.readlines()
	f.close()
	f = open(landMarkFiles[i].split()[0],'r')
	landmarks = f.readlines()
	f.close()
	data = np.zeros((len(faceRegions),1,68*2))
	
	for j in range(0,len(landmarks),2) :
		k = j / 2
		part1 = faceRegions[k].split()
		x_o,y_o,w,h = float(part1[1]),float(part1[2]),float(part1[3]),float(part1[4])
		#w,h = w+1,h+1
		data_t = []
		part2 = landmarks[j+1].split()
		for ipart in range(0,len(part2),2) :
			data_t.append((float(part2[ipart])-x_o)/w)
			#data_t.append(float(part2[ipart]))
		for ipart in range(0,len(part2),2) :
			data_t.append((float(part2[ipart+1])-y_o)/h)
			#data_t.append(float(part2[ipart+1]))
		data[k][0] = np.array(data_t)
	part = landMarkFiles[i].split("/")
	filesave = part[-1][:-4] 
	np.save(folderSave + filesave+ 'npy', data)
	np.save(folderSave + filesave +'region.npy' , np.array([x_o,y_o,w,h]))
		
	#break
	
'''
img = cv2.imread(lines[0].split()[0])
keypoints = lines[1].split()
for i in range(0,len(keypoints),2):
	cv2.circle(img,(int(float(keypoints[i])),int(float(keypoints[i+1]))),3,(255,0,0),1)
	cv2.putText(img,str(i/2+1),(int(float(keypoints[i])),int(float(keypoints[i+1]))),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
plt.imshow(img);plt.show()

'''
