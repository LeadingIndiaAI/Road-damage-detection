import darknet

class section(object):
	def __init__(self, type, options):
		self.type = type
		self.options = options # list of dict : [{key, value},{},..]

def parse_network_cfg(cfg_path):
	# list of section
	sections = read_cfg(cfg_path)

	if sections[0].type != '[net]':
		print 'Error::cfg first section must be [net], %s' %(sections[0].type)
		exit()

	net = darknet.network()
	parse_net_options(sections[0].options, net)

	params = darknet.network()
	params.h = net.h
	params.w = net.w
	params.c = net.c
	params.inputs = net.inputs
	params.batch = net.batch

	# loop all section except the first one
	net.layers = []
	for s in sections:
		if s.type == '[net]':
			continue
		options = s.options

		# keras model creation
		if s.type == '[crop]':
			l=parse_corp(options, params)
		elif s.type == '[convolutional]':
			l=parse_convolutional(options, params)
		elif s.type == '[maxpool]':
			l=parse_maxpool(options, params)
		elif s.type == '[connected]':
			l=parse_connected(options, params)
		elif s.type == '[dropout]':
			l=parse_dropout(options, params)
		elif s.type == '[detection]':
			l=parse_detection(options, params)
		else:
			print 'Error::Type not recognized: %s' %(s.type)
			exit()

		print s.type
		l.type = s.type
		net.layers.append(l)
		params.h = l.outh
		params.w = l.outw
		params.c = l.outc
		params.inputs = l.outputs
	return net

def parse_detection(options, params):
	detection_layer = darknet.abs_layer()
	detection_layer.coords = option_find_int(options, "coords", 1);
	detection_layer.classes = option_find_int(options, "classes", 1);
	detection_layer.rescore = option_find_int(options, "rescore", 0);
	#only 1 box for 1 cell
	detection_layer.num = 1 # option_find_int(options, "num", 1);
	detection_layer.side = option_find_int(options, "side", 7);
	
	# assert side*side*((1+coords)*num+classes) == inputs
	# this is the case : num's boundingbox for 1 grid cell
	# coords are for x,y,w,h, 1 is for confidence, num is for number of bounding box in 1 grid cell
	if detection_layer.side**2*((1+detection_layer.coords)*detection_layer.num+detection_layer.classes) != params.inputs:
		print 'Error::detection layer param setting is wrong'
		print 'side=%d, classes=%d, coords=%d, num=%d' %(detection_layer.side,detection_layer.classes,detection_layer.coords, detection_layer.num)
		exit()
	
	detection_layer.outputs = params.inputs
	# truth tabel is 1 boundingBox for 1 grid cell
	detection_layer.truths = detection_layer.side**2*(1+detection_layer.coords+detection_layer.classes)

	detection_layer.softmax = option_find_int(options, "softmax", 0);
	detection_layer.sqrt = option_find_int(options, "sqrt", 0);

	detection_layer.coord_scale = option_find_float(options, "coord_scale", 1);
	detection_layer.forced = option_find_int(options, "forced", 0);
	detection_layer.object_scale = option_find_float(options, "object_scale", 1);
	detection_layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
	detection_layer.class_scale = option_find_float(options, "class_scale", 1);
	detection_layer.jitter = option_find_float(options, "jitter", .2);

	# no use , just for no error
	detection_layer.outh = params.h
	detection_layer.outw = params.w
	detection_layer.outc = params.c

	return detection_layer;

def parse_dropout(options, params):
	dropout_layer = darknet.abs_layer()
	dropout_layer.probability = option_find_float(options, "probability", .5)
	dropout_layer.outw = params.w
	dropout_layer.outh = params.h
	dropout_layer.outc = params.c
	dropout_layer.outputs = params.inputs
	return dropout_layer

def parse_connected(options, params):
	connected_layer = darknet.abs_layer()
	connected_layer.outputs = option_find_int(options, "output",1)
	connected_layer.inputs = params.inputs
	connected_layer.activation_s = option_find_str(options, "activation", "logistic")

	#*weights = option_find_str(options, "weights", 0)
	#*biases = option_find_str(options, "biases", 0)
	#parse_data(biases, layer.biases, output)
	#parse_data(weights, layer.weights, params.inputs*output)

        # no use , just for no error
        connected_layer.outh = params.h
        connected_layer.outw = params.w
        connected_layer.outc = params.c

	return connected_layer

def parse_maxpool(options, params):
	maxout_layer = darknet.abs_layer()
	maxout_layer.stride = option_find_int(options, "stride",1)
	maxout_layer.size = option_find_int(options, "size",maxout_layer.stride)

	maxout_layer.h = params.h
	maxout_layer.w = params.w
	maxout_layer.c = params.c
	maxout_layer.outh = (params.h-1)/maxout_layer.stride + 1
	maxout_layer.outw = (params.w-1)/maxout_layer.stride + 1
	maxout_layer.outc = maxout_layer.c
	maxout_layer.outputs = maxout_layer.outh*maxout_layer.outw*maxout_layer.outc
	maxout_layer.batch=params.batch
	if params.h==0 or params.w==0 or params.c==0:
		print("Layer before maxpool layer must output image.")
		exit()

	return maxout_layer

def parse_convolutional(options, params):
	covl_layer = darknet.abs_layer()
	covl_layer.n = option_find_int(options, "filters",1)
	covl_layer.size = option_find_int(options, "size",1)
	covl_layer.stride = option_find_int(options, "stride",1)
	covl_layer.pad = option_find_int(options, "pad",0)
	covl_layer.activation_s = option_find_str(options, "activation", "logistic")

	covl_layer.h = params.h
	covl_layer.w = params.w
	covl_layer.c = params.c
	covl_layer.outh = convolutional_out_height(covl_layer)
	covl_layer.outw = convolutional_out_width(covl_layer)
	covl_layer.outc = covl_layer.n
	covl_layer.outputs = covl_layer.outh*covl_layer.outw*covl_layer.outc
	covl_layer.batch=params.batch
	if params.h==0 or params.w==0 or params.c==0:
		print("Layer before convolutional layer must output image.")
		exit()
	covl_layer.batch_normalize = option_find_int(options, "batch_normalize", 0)

	#layer.flipped = option_find_int_quiet(options, "flipped", 0)
	#weights = option_find_str(options, "weights", 0)
	#biases = option_find_str(options, "biases", 0)
	#parse_data(weights, layer.filters, c*n*size*size)
	#parse_data(biases, layer.biases, n)
	return covl_layer

def convolutional_out_height(l):
	if l.pad == 1:
		return l.h/l.stride+1
	else:
		return (l.h-2)/l.stride+1

def convolutional_out_width(l):
        if l.pad == 1:
                return l.w/l.stride+1
        else:
                return (l.w-2)/l.stride+1

def parse_corp(options, params):
	corp_layer = darknet.abs_layer()
	corp_layer.crop_height = option_find_int(options, "crop_height",1)
	corp_layer.crop_width = option_find_int(options, "crop_width",1)
	corp_layer.flip = option_find_int(options, "flip",0)
	corp_layer.angle = option_find_float(options, "angle",0)
	corp_layer.saturation = option_find_float(options, "saturation",1)
	corp_layer.exposure = option_find_float(options, "exposure",1)

	corp_layer.h = params.h
	corp_layer.w = params.w
	corp_layer.c = params.c
	corp_layer.outh = corp_layer.crop_height
	corp_layer.outw = corp_layer.crop_width
	corp_layer.outc = corp_layer.c
	corp_layer.batch = params.batch
	corp_layer.inputs = corp_layer.h*corp_layer.w*corp_layer.c
	corp_layer.outputs = corp_layer.outh*corp_layer.outw*corp_layer.outc
	if params.h==0 or params.w==0 or params.c==0:
		print "Layer before crop layer must output image."
		exit()

	corp_layer.noadjust = option_find_int(options, "noadjust",0)

	corp_layer.shift = option_find_float(options, "shift", 0)
	corp_layer.noadjust = corp_layer.noadjust
	return corp_layer

def option_find_int(options, opstr, defv):
	for opt in options:
		k = opt.keys()[0]
		v = opt.values()[0]
		if k == opstr:
			return int(v)
	return defv
def option_find_float(options, opstr, defv):
	for opt in options:
		k = opt.keys()[0]
		v = opt.values()[0]
		if k == opstr:
			return float(v)
	return defv

def option_find_str(options, opstr, defs):
        for opt in options:
                k = opt.keys()[0]
                v = opt.values()[0]
                if k == opstr:
                        return (v)
        return defs

def parse_net_options(options, net):
	net.batch = option_find_int(options, 'batch', 1)
	net.learning_rate = option_find_float(options, 'learning_rate', 0.001)
	net.momentum = option_find_float(options, "momentum", .9)
	net.decay = option_find_float(options, "decay", .0001)
	subdivs = option_find_int(options, "subdivisions",1)
	net.batch = net.batch / subdivs
	net.subdivisions = subdivs

	net.h = option_find_int(options, "height",0)
	net.w = option_find_int(options, "width",0)
	net.c = option_find_int(options, "channels",0)
	net.inputs = option_find_int(options, "inputs", net.h * net.w * net.c)

	if net.inputs == 0:
		print 'No input parameters supplied'
		exit()

	net.max_batches = option_find_int(options, "max_batches", 0)
	net.policy = option_find_str(options, "policy", "constant")
	if net.policy == 'step':
		net.step = option_find_int(options, "step", 1)
		net.scale = option_find_float(options, "scale", 1)
	elif net.policy == 'steps':
		l = option_find_str(options, "steps", "")
		p = option_find_str(options, "scales", "")
		if len(l)==0 or len(p)==0 :
			print 'STEPS policy must have steps and scales in cfg file'
			exit()
		ll = l.split(',')
		pp = p.split(',')
		steps = []
		scales = []
		for i in range(len(ll)):
			steps.append(int(ll[i]))
			scales.append(float(pp[i]))
		net.steps = steps
		net.scales = scales
		net.num_steps = len(ll)

def read_cfg(cfg_path):
	sections = []
	f = open(cfg_path)
	for line in f:
		if line[0] == '[':
			current = section(line.strip(), [])
			sections.append(current)
		elif len(line)>2 and line[0]!='#':
			current.options.append(read_option(line))
	return sections

def read_option(line):
	option = {}
	try:
		kv = line.strip().split('=')
		option[kv[0].strip()] = kv[1].strip()
		return option
	except:
		print 'Error::read_option %s' %(line)
