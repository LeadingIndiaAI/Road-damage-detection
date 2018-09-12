import sys
import os
sys.path.append(os.path.abspath("/src"))
import darknet
import utils
import parse
import kerasmodel
import yolodata
import ddd
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
from keras import backend as K
import keras.optimizers as opt
import cfgconst
#import opcv
import cv2
import scipy.misc
import tensorflow as tf
import keras
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.callbacks import EarlyStopping, ModelCheckpoint

# define constant

#cpu config
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


det_l = cfgconst.net.layers[len(cfgconst.net.layers)-1]
CLASSNUM = det_l.classes
f = open(cfgconst.labelnames)
voc_names =[]
for ln in f:
	voc_names.append(ln.strip()) # = ["stopsign", "skis"]

# check class number
print voc_names
if CLASSNUM != len(voc_names):
	print 'cfg file class setting is not equal to '+cfgconst.labelnames
	exit()

# run_yolo

if len(sys.argv) < 2:
	print ('usage: python %s [train/test/valid] [pretrained model (optional)]\n' %(sys.argv[0]))
	exit()

voc_labels= []
for i in range(CLASSNUM):
	voc_labels.append("ui_data/labels/"+voc_names[i]+".PNG")
	if  not os.path.isfile(voc_labels[i]):
		print ('can not load image:%s' %(voc_labels[i]))
		exit()


import utils
thresh = utils.find_float_arg(sys.argv, "-thresh", .2)
#print 'thresh='+str(thresh)
#exit()
cam_index = utils.find_int_arg(sys.argv, "-c", 0)
#cfg_path = sys.argv[2]
model_weights_path = sys.argv[2] if len(sys.argv) > 2 else 'noweight'
filename = sys.argv[3] if len(sys.argv) > 3 else 'nofilename'

print sys.argv
print model_weights_path+','+filename

def train_yolo( weights_path):


	# construct network
	net = cfgconst.net  #parse.parse_network_cfg(cfg_path)
	train_images = cfgconst.train  #"train_data/train.txt"
	backup_directory = "backup/"

	# load pretrained model 
	if os.path.isfile(model_weights_path):
		print 'Loading '+model_weights_path
		model=load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
		sgd = opt.SGD(lr=net.learning_rate, decay=net.decay, momentum=net.momentum, nesterov=True)
		model.compile(loss=ddd.yololoss, optimizer=sgd, metrics=["accuracy"])

	else:
	
		# base is cfg name
		#base = utils.basecfg(cfg_path)

		print ('Learning Rate: %f, Momentum: %f, Decay: %f\n' %(net.learning_rate, net.momentum, net.decay));
		model = kerasmodel.makenetwork(net)

	(X_train, Y_train) = yolodata.load_data(train_images,net.h,net.w,net.c, net)

	print ('max_batches : %d, X_train: %d, batch: %d\n' %(net.max_batches, len(X_train), net.batch));
	print str(net.max_batches/(len(X_train)/net.batch))

	#datagen = ImageDataGenerator(
	#	featurewise_center=True,
	#	featurewise_std_normalization=True,
	#	rotation_range=0,
	#	width_shift_range=0.,
	#	height_shift_range=0.,
	#	horizontal_flip=True)

	#datagen.fit(X_train)

	#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=net.batch),
        #            samples_per_epoch=len(X_train), nb_epoch=net.max_batches/(len(X_train)/net.batch))
	#model.fit(X_train, Y_train, batch_size=net.batch, nb_epoch=net.max_batches/(len(X_train)/net.batch))
	early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

	checkpoint = ModelCheckpoint('yolo_weight.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)
	batchesPerdataset = max(1,len(X_train)/net.batch)
	model.fit(X_train, Y_train, nb_epoch=net.max_batches/(batchesPerdataset), batch_size=net.batch, verbose=1)

	model.save_weights('yolo_weight_rd.h5')
	model.save('yolo_kerasmodel_rd.h5')

def debug_yolo( cfg_path, model_weights_path='yolo_kerasmodel_rd.h5' ):
	net = cfgconst.net ##parse.parse_network_cfg(cfg_path)
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape
	x_test,y_test = yolodata.load_data('train_data/test.txt', h, w, c, net)
	testloss = testmodel.evaluate(x_test,y_test)
	print y_test
	print 'testloss= '+str(testloss)


def predict(X_test, testmodel, confid_thresh):
	print 'predict, confid_thresh='+str(confid_thresh)
	
	pred = testmodel.predict(X_test)
	(s,w,h,c) = testmodel.layers[0].input_shape
	
	# find confidence value > 0.5
	confid_index_list =[]
	confid_value_list =[]
	x_value_list = []
	y_value_list =[]
	w_value_list =[]
	h_value_list =[]
	class_id_list =[]
	classprob_list =[]
	x0_list = []
	x1_list = []
	y0_list = []
	y1_list = []
	det_l = cfgconst.net.layers[len(cfgconst.net.layers)-1]
        side = det_l.side
	classes = det_l.classes
	xtext_index =0
	foundindex = False
	max_confid =0
	#
	for p in pred:
		#foundindex = False
		for k in range(1): #5+classes):
			#print 'L'+str(k)
			for i in range(side):
				for j in range(side):
					if k==0:
						max_confid = max(max_confid,p[k*49+i*7+j])

					#sys.stdout.write( str(p[k*49+i*7+j])+', ' )
					if k==0 and p[k*49+i*7+j]>confid_thresh:
						confid_index_list.append(i*7+j)
						foundindex = True
				#print '-'
		print 'max_confid='+str(max_confid)
		#
		for confid_index in confid_index_list:
			confid_value = max(0,p[0*49+confid_index])
			x_value = max(0,p[1*49+confid_index])
			y_value = max(0,p[2*49+confid_index])
			w_value = max(0,p[3*49+confid_index])
			h_value = max(0,p[4*49+confid_index])
			maxclassprob = 0
			maxclassprob_i =-1
			for i in range(classes):
				if p[(5+i)*49+confid_index] > maxclassprob and foundindex:
					maxclassprob = p[(5+i)*49+confid_index]
					maxclassprob_i = i

			classprob_list.append( maxclassprob)
			class_id_list.append( maxclassprob_i)

			print 'max_confid='+str(max_confid)+',c='+str(confid_value)+',x='+str(x_value)+',y='+str(y_value)+',w='+str(w_value)+',h='+str(h_value)+',cid='+str(maxclassprob_i)+',prob='+str(maxclassprob)
		#
			row = confid_index / side
			col = confid_index % side
			x = (w / side) * (col + x_value)
			y = (w / side) * (row + y_value)

			print 'confid_index='+str(confid_index)+',x='+str(x)+',y='+str(y)+',row='+str(row)+',col='+str(col)

		#draw = ImageDraw.Draw(nim)
		#draw.rectangle([x-(w_value/2)*w,y-(h_value/2)*h,x+(w_value/2)*w,y+(h_value/2)*h])
		#del draw
		#nim.save('predbox.png')
		
		#sourceimage = X_test[xtext_index].copy()

			x0_list.append( max(0, int(x-(w_value/2)*w)) )
			y0_list.append( max(0, int(y-(h_value/2)*h)) )
			x1_list.append( int(x+(w_value/2)*w) )
			y1_list.append( int(y+(h_value/2)*h) )
		
		break
		#xtext_index = xtext_index + 1

	#print pred
	sourceimage = X_test[0].copy()
	return sourceimage, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list


def test_yolo(imglist_path, model_weights_path='yolo_kerasmodel_rd.h5', confid_thresh=0.3):
        print 'test_yolo: '+imglist_path
        # custom objective function
        #print (s,w,h,c)
        #exit()
	if os.path.isfile(imglist_path):
        	testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
        	(s,w,h,c) = testmodel.layers[0].input_shape
		f = open(imglist_path)
		for img_path in f:
		#
        		#X_test = []
        		if os.path.isfile(img_path.strip()):
                		frame = Image.open(img_path.strip())
                		#(orgw,orgh) = img.size
				nim = scipy.misc.imresize(frame, (w, h, c))
				if nim.shape != (w, h, c):
					continue

                		#nim = img.resize( (w, h), Image.BILINEAR )
				img, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list = predict(np.asarray([nim]), testmodel, thresh)
                		#X_test.append(np.asarray(nim))
        			#predict(np.asarray(X_test), testmodel, confid_thresh)
				# found confid box
                		for x0,y0,x1,y1,classprob,class_id in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list):
                		#
					# draw bounding box
					cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
					# draw classimg
					classimg = cv2.imread(voc_labels[class_id])
					print 'box='+str(x0)+','+str(y0)+','+str(x1)+','+str(y1)
					#print img.shape
					#print classimg.shape
					yst = max(0,y0-classimg.shape[0])
					yend = max(y0,classimg.shape[0])
					img[yst:yend, x0:x0+classimg.shape[1]] = classimg
					# draw text
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, str(classprob), (x0,y0-classimg.shape[0]-1), font, 1,(255,255,255),2,cv2.LINE_AA)
					#
				cv2.imshow('frame',img)
				if cv2.waitKey(1000) & 0xFF == ord('q'):
					break


			else:
				print img_path+' predict fail'

		cv2.destroyAllWindows()
	else:
		print imglist_path+' does not exist'
	


def demo_yolo(model_weights_path, filename, thresh=0.3):
	print 'demo_yolo'
	testmodel = load_model(model_weights_path, custom_objects={'yololoss': ddd.yololoss})
	(s,w,h,c) = testmodel.layers[0].input_shape

	cap = cv2.VideoCapture(filename)

	while (cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			break
		#print frame
		nim = scipy.misc.imresize(frame, (w, h, c))
		#nim = np.resize(frame, (w, h, c)) #, Image.BILINEAR )

		img, x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list = predict(np.asarray([nim]), testmodel, thresh)
		# found confid box
		for x0,y0,x1,y1,classprob,class_id in zip(x0_list, y0_list, x1_list, y1_list, classprob_list, class_id_list): 
		#
			# draw bounding box
			cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 2)
			# draw classimg
			classimg = cv2.imread(voc_labels[class_id])
			print 'box='+str(x0)+','+str(y0)+','+str(x1)+','+str(y1)
			#print img.shape
			#print classimg.shape
			yst = max(0,y0-classimg.shape[0])
			yend = max(y0,classimg.shape[0])
			img[yst:yend, x0:x0+classimg.shape[1]] = classimg
			# draw text
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img, str(classprob), (x0,y0-classimg.shape[0]-1), font, 1,(255,255,255),2,cv2.LINE_AA)
			#
		cv2.imshow('frame',img)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if sys.argv[1]=='train':
        train_yolo(model_weights_path)
elif sys.argv[1]=='test':
	if os.path.isfile(model_weights_path):
        	test_yolo(filename, model_weights_path, confid_thresh=thresh)
	else:
		test_yolo(filename, confid_thresh=thresh)
elif sys.argv[1]=='demo_video':
	if os.path.isfile(model_weights_path):
		print 'pretrain model:'+model_weights_path+', video:'+filename+', thresh:'+str(thresh)
        	demo_yolo(model_weights_path, filename, thresh)
	else:
		print 'syntax error::need specify a pretrained model'
		exit()
elif sys.argv[1]=='debug':
        debug_yolo( cfg_path, model_weights_path )

