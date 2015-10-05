import sys
import glob
import os
import shutil
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

datadir = sys.argv[1]
pcadir = sys.argv[2]
modeldir = sys.argv[3]
used = int(sys.argv[4])
model = sys.argv[5]
step = 50

ipca = joblib.load("%s/pca.pkl"%pcadir)

def load_data(start, end, step):
	data = []
	labels = []
	for batch_start in xrange(start, end, step): 
		batch_end = min(batch_start+step, end)
		print "Loading from %d to %d" % (batch_start,batch_end)
		partial_data = []
		for fname in fnames[batch_start:batch_end]:	
			print "Loading image", fname
			for angle in xrange(0,360,5):
				partial_data.append(np.loadtxt("%s/%s-%d.txt" % (datadir,fname,angle), dtype=int))
				labels.append(angle)
		partial_data = ipca.transform(partial_data).tolist()
		data += partial_data
	return data, labels

fnames = [os.path.basename(filename)[:-6] for filename in glob.glob("%s/*-0.txt" % datadir)]
fnames = np.random.permutation(fnames)

data, labels = load_data(0, used, step)

if model == "SVR":
	print "Using SVR"
	clf = svm.LinearSVR()
if model == "DF":
	print "Using DF"
	clf = RandomForestClassifier(n_estimators=10)
else:
	clf = svm.LinearSVC()
clf.fit(data, labels)  

if os.path.isdir(modeldir):
	shutil.rmtree(modeldir)
os.mkdir(modeldir)
joblib.dump(clf, "%s/clf.pkl"%modeldir)

