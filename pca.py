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
pcadir = sys.argv[2]
n_comp = 300
step = 1000 

fnames = [os.path.basename(filename) for filename in glob.glob("%s/*.txt" % datadir)]

used = len(fnames)

ipca = IncrementalPCA(n_components=n_comp)

for batch_start in xrange(0, used, step):
	batch_end = min(batch_start+step, used)
	print "Loading from %d to %d" % (batch_start,batch_end)
	data = []
	labels = []
	for fname in fnames[batch_start:batch_end]:	
		print "Loading image", fname
		data.append(np.loadtxt("%s/%s" % (datadir,fname)))
	print np.array(data).shape
	ipca.partial_fit(data)


if os.path.isdir(pcadir):
	shutil.rmtree(pcadir)
os.mkdir(pcadir)	
joblib.dump(ipca, "%s/pca.pkl"%pcadir)
print np.sum(ipca.explained_variance_ratio_)

