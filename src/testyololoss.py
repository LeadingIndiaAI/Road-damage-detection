import theano
import theano.tensor as T
import numpy
import tensorflow as tf
from keras import backend as K

DDDEBUG = False
def check(detection_layer,model):
	detection_layer = detection_layer
	bnumPercell = detection_layer.num
	bckptsPercell = 1+detection_layer.coords
	totbkptsPercell = bckptsPercell * detection_layer.num
	predsofcell = totbkptsPercell + detection_layer.classes
	lamda_coor = detection_layer.coord_scale
	lamda_noobject = detection_layer.noobject_scale
	truthsofcell = predsofcell
	gridcells = detection_layer.side
	expected = gridcells**2*(predsofcell)
	real = model.layers[len(model.layers)-1].output_shape[1]
	if expected != real:
		print 'cfg detection layer setting mismatch::change cfg setting'
		print 'output layer should be '+str(gridcells**2*(predsofcell))
		print 'actual output layer is '+str(model.layers[len(model.layers)-1].output_shape[1])
		exit()

def yololoss(y_true, y_pred):
	gridcells = 7
	bnumPercell = 1
	bckptsPercell = 5
	totbkptsPercell = bckptsPercell * bnumPercell
	predsofcell = totbkptsPercell + 2 # classes is 2
	lamda_coor = 48
	lamda_noobject = 1
	lamda_xy = 1
	lamda_wh = 1
	truthsofcell = predsofcell
	print 'check these values with cfg setting, if is not correct , must edit yololoss() func code'
	print 'gridcells='+str(gridcells)+', bnumPercell='+str(bnumPercell)+', classes=2'
#
	# truth table format is [[confid,x,y,w,h]..,classes] for one cell
	totloss =0
	for cell in range(gridcells**2):
		confidloss =0
		xyloss =0
		whloss =0
		for i in range(bnumPercell):
			confidloss += (y_true[cell*truthsofcell:cell*truthsofcell+1] - y_pred[(cell*predsofcell)+(i*bckptsPercell):(cell*predsofcell+1)+(i*bckptsPercell)])**2
			xyloss += (y_true[cell*truthsofcell+1:cell*truthsofcell+2+1] - y_pred[(cell*predsofcell)+(i*bckptsPercell)+1:(cell*predsofcell)+(i*bckptsPercell)+2+1])**2
			whloss += (tf.sqrt(y_true[cell*truthsofcell+2+1:cell*truthsofcell+4+1]) - tf.sqrt(y_pred[(cell*predsofcell+2)+(i*bckptsPercell)+1:(cell*predsofcell+4+1)+(i*bckptsPercell)]))**2
	
		#
		porbloss = (y_true[cell*truthsofcell+bnumPercell*totbkptsPercell:(cell+1)*truthsofcell] - y_pred[cell*predsofcell+(bnumPercell*totbkptsPercell):(cell+1)*predsofcell])**2

		#
		t = K.greater(y_true[cell*truthsofcell], tf.constant(0.5))
		confidloss_conditioned = tf.select(t, lamda_coor*(confidloss), lamda_noobject*(confidloss))
		sumconfidloss = K.sum(confidloss_conditioned)

		#if y_true[cell*truthsofcell] == 1 :  # refer to confidence, I OBJ ij, reference yolo paper
		#if t is not None:  # refer to confidence, I OBJ ij, reference yolo paper
		#	sumxyloss = lamda_xy*tf.reduce_sum(xyloss)
		#	sumwhloss = lamda_wh*tf.reduce_sum(whloss)
		#	sumporbloss = tf.reduce_sum(porbloss)
		#	sumconfidloss = lamda_coor*tf.reduce_sum(confidloss)
		#else:
		#	sumxyloss = 0 #tf.reduce_sum(xyloss) #0
		#	sumwhloss = 0 #tf.reduce_sum(whloss) #0
		#	sumporbloss = tf.reduce_sum(porbloss) #0
		#	sumconfidloss = lamda_noobject*tf.reduce_sum(confidloss)

		#totloss += lamda_coor*(sumxyloss+sumwhloss)+sumconfidloss+sumporbloss
		#totloss += sumxyloss+sumwhloss+sumconfidloss+sumporbloss
		totloss += sumconfidloss

	return totloss

#
# 10 examples  
# 
if DDDEBUG:
	xind = numpy.random.random((10, 20))
	y_true = numpy.random.random((10, 20))

	yt = T.matrix('yt')
	xin = T.matrix('xin')

	yp = 2*xin
	loss = yolo_loss(yt,yp)

	gw = T.grad(loss, wrt=xin)
	f = theano.function([yt,xin], loss)
	f1 = theano.function([yt,xin], gw)

	print f(y_true,xind)
	print f1(y_true,xind)
