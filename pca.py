import sys
import glob
import os
import shutil
import numpy as np
import math

from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.externals import joblib

datadir = sys.argv[1]
angle_step = int(sys.argv[2])
pcadir = sys.argv[3]
used = int(sys.argv[4])
n_comp = 300
step = 30 

if step*360.0/angle_step <= n_comp:   # for some reason if the number of samples per iteration is less than n_components it breaks
	step = int(math.ceil(n_comp/(360.0/angle_step))) + 10

print "Step:", step

fnames = [os.path.basename(filename)[:-6] for filename in glob.glob("%s/*-0.txt" % datadir)]
fnames = np.random.permutation(fnames)

ipca = IncrementalPCA(n_components=n_comp)

for batch_start in xrange(0, used, step):
	batch_end = min(batch_start+step, used)
	print "Loading from %d to %d" % (batch_start,batch_end)
	data = []
	labels = []
	for fname in fnames[batch_start:batch_end]:	
		print "Loading image", fname
		for angle in xrange(0,360,angle_step):
			data.append(np.loadtxt("%s/%s-%d.txt" % (datadir,fname,angle)))
			labels.append(angle)
	print np.array(data).shape
	ipca.partial_fit(data)


if os.path.isdir(pcadir):
	shutil.rmtree(pcadir)
os.mkdir(pcadir)	
joblib.dump(ipca, "%s/pca.pkl"%pcadir)
print np.sum(ipca.explained_variance_ratio_)

