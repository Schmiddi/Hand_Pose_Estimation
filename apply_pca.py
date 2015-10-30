import sys
import glob
import os
import shutil
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

srcdir = sys.argv[1]
pcadir = sys.argv[2]
destdir = sys.argv[3]

ipca = joblib.load("%s/pca.pkl"%pcadir)

fnames = [os.path.basename(filename) for filename in glob.glob("%s/*.txt" % srcdir)]

for fname in fnames:	
	print "Loading image", fname
	partial_data = ipca.transform([np.loadtxt("%s/%s" % (srcdir,fname), dtype=int)]).tolist()
	np.savetxt("%s/%s" % (destdir,fname), partial_data[0])


