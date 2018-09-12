import numpy as np
import cv2
import time

def showboximage(img, x0, y0, x1, y1, classprob, classimgpath, frameTime=100):
	classimg = cv2.imread(classimgpath)
	# draw bounding box
	cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
	# draw classimg
	img[y0-classimg.shape[0]:y0, x0:x0+classimg.shape[1]] = classimg
	# draw text
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, str(classprob), (x0,y0-classimg.shape[0]-1), font, 4,(255,255,255),2,cv2.LINE_AA)

	cv2.imshow('frame',img)
	time.sleep(frameTime/1000.) # delays for frameTime milliseconds

