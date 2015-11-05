import sys
import glob
import os
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.externals import joblib

datadir = sys.argv[1]
pcadir = sys.argv[2]
modeldir = sys.argv[3]
used = int(sys.argv[4])
step = 50

ipca = joblib.load("%s/pca.pkl"%pcadir)
clf = joblib.load("%s/clf.pkl"%modeldir)

def load_data(start, end, step):
	data = []
	labels = []
	for batch_start in xrange(start, end, step): 
		batch_end = min(batch_start+step, end)
		print "Loading from %d to %d" % (batch_start,batch_end)
		partial_data = []
		for fname in fnames[batch_start:batch_end]:
			print "Loading image", fname
			partial_data.append(np.loadtxt("%s/%s" % (datadir,fname)))
			angle = int(fname.split('-')[1].split('.')[0])
			labels.append(angle)
		partial_data = ipca.transform(partial_data).tolist()
		data += partial_data
	return data, labels

fnames = [os.path.basename(filename) for filename in glob.glob("%s/*.txt" % datadir)]
fnames = np.random.permutation(fnames)

data, labels = load_data(0, used, step)

predict = clf.predict(data)

tot_diff = 0.0
errors = []
for i in xrange(0,len(labels)):
	angle_diff = labels[i]-predict[i]
	if angle_diff > 180: angle_diff -= 360 
	if angle_diff < -180: angle_diff += 360 
	tot_diff += abs(angle_diff)	
	errors.append(abs(angle_diff))

fres = open("angle-errors", "w")
for i in range(used):
	fres.write(fnames[i] + " " + str(predict[i]) + "\n")
print "Average error is %.2f degrees" % (tot_diff/len(labels))

