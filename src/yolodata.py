from PIL import Image, ImageDraw
import numpy as np
import sys

class yolobbox():
	def __init__(self):
		self

def readlabel(fn):
	#print 'readlabel '+ fn
	boxlist = []
	f = open(fn)
	box = yolobbox()
	for l in f:
		try:
			ss= l.split(' ')
			box.id = int(ss[0])
			box.x = float(ss[1]) 
			box.y = float(ss[2]) 
			box.w = float(ss[3]) 
			box.h = float(ss[4]) 
		except:
			box.id = -1
		boxlist.append(box)
	return boxlist
		
def load_data(train_images, h, w, c, net):
	f = open(train_images)
	paths = []
	for l in f:
		paths.append(l)

	bckptsPercell = net.layers[len(net.layers)-1].coords + 1
	gridcells = net.layers[len(net.layers)-1].side 
	bnumPercell = net.layers[len(net.layers)-1].num
	classes = net.layers[len(net.layers)-1].classes

	X_train = []
	Y_train = []
	count = 1
	for fn in paths:
		img = Image.open( fn.strip())
		(orgw,orgh) = img.size
		nim = img.resize( (w, h), Image.BILINEAR )
		data = np.asarray( nim )
		if data.shape != (w, h, c):
			continue
		X_train.append(data)

		# replace to label path
		fn=fn.replace("/images/","/labels/")
		fn=fn.replace("/JPEGImages/","/labels/")  #VOC
		fn=fn.replace(".JPEG",".txt")
		fn=fn.replace(".jpg",".txt")              #VOC
		#fn=fn.replace(".JPG",".txt")
		#print fn

		#
		# may have multi bounding box for 1 image
		boxlist = readlabel(fn.strip())
		for box in boxlist:
			if box.id == -1:
				print 'read bbox fail'
				continue


			#
			# let truth size == pred size, different from yolo.c 
			# trurh data arrangement is (confid,x,y,w,h)(..)(classes)
			#
			truth = np.zeros(gridcells**2*(bckptsPercell*bnumPercell+classes))
			col = int(box.x * gridcells)
			row = int(box.y * gridcells)
			x = box.x * gridcells - col
			y = box.y * gridcells - row
			#print((5+box.id)*(gridcells**2)+index)
			# only 1 box for 1 cell
			#for i in range(bnumPercell):
			index = (col+row*gridcells)
			truth[index] = 1
			truth[gridcells**2+index] = x
			truth[2*(gridcells**2)+index] = y
			truth[3*(gridcells**2)+index] = box.w
			truth[4*(gridcells**2)+index] = box.h
			#print 'index='+str(index)+' '+str(box.x)+' '+str(box.y)+' '+str(box.w)+' '+str(box.h)
			if ((5+box.id)*(gridcells**2)+index)>637:
				print(fn)
			truth[(5+box.id)*(gridcells**2)+index] =1

		#
		Y_train.append(truth)

		#print 'draw rect bounding box'
		#draw = ImageDraw.Draw(img)
		#draw.rectangle([(box.x-box.w/2)*orgw,(box.y-box.h/2)*orgh,(box.x+box.w/2)*orgw,(box.y+box.h/2)*orgh])
		#del draw
		#img.save('ttt.png')
		#exit()
		#for k in range(7):
		#	print 'L'+str(k)
		#	for row_cell in range(7):
		#		for col_cell in range(7):
		#			sys.stdout.write( str(truth[k*49+col_cell+row_cell*(7)])+', ' )
		#		print '-'

		#print truth[720:740]
		#exit()
		# this is for debug
		if count > 100000:
			break
		else:
			count = count + 1

	#print len(X_train)
	XX_train = np.asarray(X_train)
	YY_train = np.asarray(Y_train)
	print XX_train.shape
	print YY_train.shape
	#exit()

	return XX_train, YY_train
		

