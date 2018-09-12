from keras import backend as K
import tensorflow as tf
import numpy as np
import cfgconst

CFGEN = True
if CFGEN:
	det_l = cfgconst.net.layers[len(cfgconst.net.layers)-1]
	side = det_l.side
        gridcells = side**2
        lamda_confid_obj = det_l.object_scale
        lamda_confid_noobj = det_l.noobject_scale
        lamda_xy = det_l.coord_scale
        lamda_wh = det_l.coord_scale
        reguralar_wh = 0
        lamda_class = det_l.class_scale
        classes = det_l.classes

else:
	gridcells = 7**2
	lamda_confid_obj = 48
	lamda_confid_noobj = 1
	lamda_xy = 10
	lamda_wh = 15
	reguralar_wh = 0.1
	lamda_class = 20
	classes = 2

DEBUG_loss = False

# shape is (gridcells,)
def yoloconfidloss(y_true, y_pred, t):
	lo = K.square(y_true-y_pred)
	value_if_true = lamda_confid_obj*(lo)
	value_if_false = lamda_confid_noobj*(lo)
	loss1 = tf.where(t, value_if_true, value_if_false)
	loss = K.mean(loss1) #,axis=0)
	#
	return loss

# shape is (gridcells*2,)
def yoloxyloss(y_true, y_pred, t):
        lo = K.square(y_true-y_pred)
        value_if_true = lamda_xy*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)
	return K.mean(loss1)

# shape is (gridcells*2,)
def yolowhloss(y_true, y_pred, t):
        #lo = K.square(K.sqrt(y_true)-K.sqrt(y_pred))
	# let w,h not too small or large
        lo = K.square(y_true-y_pred)+reguralar_wh*K.square(0.5-y_pred)
        value_if_true = lamda_wh*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)
	return K.mean(loss1)

# shape is (gridcells*classes,)
def yoloclassloss(y_true, y_pred, t):
        lo = K.square(y_true-y_pred)
        value_if_true = lamda_class*(lo)
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)
	return K.mean(loss1)

# shape is (gridcells*(5+classes), )
def yololoss(y_true, y_pred):
        truth_confid_tf = tf.slice(y_true, [0,0], [-1,gridcells])
        truth_x_tf = tf.slice(y_true, [0,gridcells], [-1,gridcells])
        truth_y_tf = tf.slice(y_true, [0,gridcells*2], [-1,gridcells])
        truth_w_tf = tf.slice(y_true, [0,gridcells*3], [-1,gridcells])
        truth_h_tf = tf.slice(y_true, [0,gridcells*4], [-1,gridcells])

	truth_classes_tf = []
	for i in range(classes):
        	ctf = tf.slice(y_true, [0,gridcells*(5+i)], [-1,gridcells])
		truth_classes_tf.append(ctf)

        pred_confid_tf = tf.slice(y_pred, [0,0], [-1,gridcells])
        pred_x_tf = tf.slice(y_pred, [0,gridcells], [-1,gridcells])
        pred_y_tf = tf.slice(y_pred, [0,gridcells*2], [-1,gridcells])
        pred_w_tf = tf.slice(y_pred, [0,gridcells*3], [-1,gridcells])
        pred_h_tf = tf.slice(y_pred, [0,gridcells*4], [-1,gridcells])

	pred_classes_tf = []
	for i in range(classes):
        	ctf = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
		pred_classes_tf.append(ctf)

	t = K.greater(truth_confid_tf, 0.5) 

	confidloss = yoloconfidloss(truth_confid_tf, pred_confid_tf, t)
	xloss = yoloxyloss(truth_x_tf, pred_x_tf, t)
	yloss = yoloxyloss(truth_y_tf, pred_y_tf, t)
	wloss = yolowhloss(truth_w_tf, pred_w_tf, t)
	hloss = yolowhloss(truth_h_tf, pred_h_tf, t)

	classesloss =0
	for i in range(classes):
		closs = yoloclassloss(truth_classes_tf[i], pred_classes_tf[i], t)
		classesloss += closs

	loss = confidloss+xloss+yloss+wloss+hloss+classesloss
	#loss = wloss+hloss
	#
	#return loss,confidloss,xloss,yloss,wloss,hloss,classesloss
	return loss



def check(detection_layer,model):
        expected = gridcells*(5+classes)
        real = model.layers[len(model.layers)-1].output_shape[1]
        if expected != real:
                print 'cfg detection layer setting mismatch::change cfg setting'
                print 'output layer should be '+str(expected)+'neurons'
                print 'actual output layer is '+str(real)+'neurons'
                exit()

#
#
if DEBUG_loss:
	x =K.placeholder(ndim=2)
	y =K.placeholder(ndim=2)
	loss,confidloss,xloss,yloss,wloss,hloss,classesloss = yololoss(y,x)

	f = K.function([y,x], [loss,confidloss,xloss,yloss,wloss,hloss,classesloss])

	xtrain = np.ones(343*10).reshape(10,343)
	ytrain = np.zeros(343*10).reshape(10,343)
	ytrain[0][0]=1
	ytrain[0][49]=0.1
	ytrain[0][49*2]=0.2
	ytrain[0][49*3]=0.3
	ytrain[0][49*4]=0.4
	ytrain[0][49*5]=1


	print f([ytrain,xtrain])

