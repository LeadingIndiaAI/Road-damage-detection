import parse
import os

f = open('workingcfg.txt')
hascfg = False
for l in f:
	ss = l.split('=')
	if ss[0].strip() == 'cfg':
		cfgpath = ss[1].strip()
	elif ss[0].strip() == 'train':
		train = ss[1].strip()
	elif ss[0].strip() == 'labelnames':
		labelnames = ss[1].strip()

#
if os.path.isfile(cfgpath):
	net = parse.parse_network_cfg(cfgpath)
else:
	print 'Error::workingcfg.txt dont contain valid cfg file'
	exit()
