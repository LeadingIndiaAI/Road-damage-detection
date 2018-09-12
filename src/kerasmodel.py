import darknet
import keras.layers.advanced_activations as a_a
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization as BNOR
from keras.layers.convolutional import Convolution2D as C2D
from keras.layers.pooling import MaxPooling2D as MP2D
from keras.layers.pooling import GlobalAveragePooling2D as GAP2D
import tensorflow as tf
import keras.optimizers as opt


# loss really use this
import ddd



def printmodel(model):
	#print len(model.layers)
	#print model.layers[31].output_shape[1]
	for l in model.layers:
		#print l.activation
		try:
			print l.name+':'+str(l.input_shape)+'-->'+str(l.output_shape)+' act = '+l.activation.name
		except:
			print l.name+':'+str(l.input_shape)+'-->'+str(l.output_shape)



def makenetwork(net):
	model = Sequential()
	index =0
	for l in net.layers:
		try:
			if l.activation_s == 'leaky' or l.activation_s == 'relu':
				#act = a_a.LeakyReLU(alpha=0.1)
				act = 'relu'
			elif l.activation_s == 'logistic' or l.activation_s == 'sigmoid':
				act = 'sigmoid'
			else:
				act = 'linear'
			print 'activation='+act
		except:
			print 'no activation at index '+str(index)

		if l.type == '[crop]':
			#model.add(Input(shape=(l.outh*l.outw*l.outc,), name='input'+str(index)))
			crop_shape = (l.outh,l.outw,l.outc,)
			insert_input = True
		elif l.type == '[convolutional]':
			if l.pad == 1:
				pad = 'same'
			else:
				pad = 'valid'
			if insert_input:
				model.add(C2D( l.n, l.size, l.size, activation=act, border_mode=pad, subsample=(l.stride,l.stride), input_shape=crop_shape, name='convol'+str(index)))
			else:
				model.add(C2D( l.n, l.size, l.size, activation=act, border_mode=pad, subsample=(l.stride,l.stride), name='convol'+str(index)))
			if l.batch_normalize == 1:
				model.add(BNOR(name='bnor'+str(index)))
			insert_input = False
		elif l.type == '[maxpool]':
			model.add(MP2D( pool_size=(l.size,l.size),strides=(l.stride,l.stride),name='maxpool'+str(index) ))
		elif l.type == '[connected]':
			try:
				model.add(Flatten(name='flattern'+str(index)))
			except:
				print 'no need to flattern'
			model.add(Dense( l.outputs, activation=act, name='conted'+str(index)))
		elif l.type == '[dropout]':
			model.add(Dropout(l.probability, name='dropout'+str(index) ))
		elif l.type == '[detection]':
			ddd.check(l,model)
			sgd = opt.SGD(lr=net.learning_rate, decay=net.decay, momentum=net.momentum, nesterov=True)
			model.compile(loss=ddd.yololoss, optimizer=sgd)
		print l.type + str(index)
		index = index+1

	printmodel(model)
	return model
