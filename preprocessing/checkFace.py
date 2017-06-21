import glob
import cv2
path = "/data5/Real_Fake_expression/data/testmetadata/"
subjects = glob.glob(path + "*.txt")

for subject in subjects:
	s = subject.split('/')[-1][:-4]
	f = open(subject,'r')
	lines = f.readlines()
	f.close()
	part = lines[2].split()
	filename,x,y,w,h = part[0],int(part[1]),int(part[2]),int(part[3]),int(part[4])
	print(filename)
	img = cv2.imread(filename)
	
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.imshow("",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	   	break

