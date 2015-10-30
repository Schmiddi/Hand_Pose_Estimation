import sys
import glob
import os
import shutil
import numpy as np
import ipdb as pdb
import operator
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import preprocessing

datadir = "ReducedFeatures/Features1x1_nod"
labeldir = "TrainData"
modeldir = "."
used = 2000
model = "SVR"

try:
    datadir = sys.argv[1]
    labeldir = sys.argv[2]
    modeldir = sys.argv[3]
    used = int(sys.argv[4])
    model = sys.argv[5]
except IndexError:
    print "Usage: hand_config_est.py datadir labeldir modeldir nooftrainingsamples model"
    print "Defaults used"

#ipca = joblib.load("%s/pca.pkl"%pcadir)

def load_data(start, end):
    """ load data from files in datadir and labeldir, from start to end
    shuffle samples randomly    
    """
    data = []
    labels = []
    for i in np.random.permutation(range(start, end)): 
        data.append(np.loadtxt("%s/%d-0.txt" % (datadir,i), dtype=float))
        raw_label = np.loadtxt("%s/%d.txt" % (labeldir,i), dtype=float, )
        labels.append(raw_label.flatten())
    return np.array(data), np.array(labels)

# load training data to Xall and labels to yall, only if variables dont exists
#try:
#    Xall
#    yall
#except NameError:
Xall, yall = load_data(0, 3000)

# normalize both x and y
#preprocessing.scale(Xall, copy=False)

scaler = preprocessing.MinMaxScaler().fit(Xall)
Xall = scaler.transform(Xall)
scaler = preprocessing.MinMaxScaler().fit(yall)
yall = scaler.transform(yall)

# this is the cross validation set, from used to end
Xcv = Xall[used+1:, :]
ycv = yall[used+1:, 1]

#plt.scatter(yall[:,1], yall[:,2])

m = used # number of training samples
C = 10.0 # C param for SVR
gamma = 0.1 # gamma param for gaussian kernel, inverse of variance
errors = []
models = []

def tune(feature):
    """
    trains SVR with different gamma and C param and chose the combinantion
    that results in lowest error on cross validation set
    """
    performance = []
    #vals = [0.1, 0.3, 1.0, 3.0, 10.0]
    vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for C in vals:
        C = C*10
        for gamma in vals:
            gamma = gamma / 10
            X = Xall[0:m,:]
            y = yall[0:m, feature]
            ycv = yall[used+1:, feature]
        
            clf = svm.SVR(C=C, gamma=gamma)
            clf.fit(X, y)
            
            pred = clf.predict(X)
            err = np.mean(np.square(np.subtract(y, pred)))
        
            predcv = clf.predict(Xcv)
            errcv = np.mean(np.square(np.subtract(ycv, predcv)))
        
            performance.append([m, C, gamma, err, errcv])
            print performance[-1]
            
    opt_dex = min(range(len(performance)), key=lambda x: performance[x][4])
    opt = performance[opt_dex]
    
    print "optimal: ", opt
    return opt[1], opt[2]

def pred_error(y, y_pred):
    #err = np.mean(abs((y-pred)/y))
    # average squared difference
#    y_raw = y*scaler.std_[1] + scaler.mean_[1]
#    ypred_raw = y_pred*scaler.std_[1] + scaler.mean_[1]

    y_raw = (y - scaler.min_[1]) / scaler.scale_[1]
    ypred_raw = (y_pred - scaler.min_[1]) / scaler.scale_[1]
    return np.mean(np.square(y_raw-ypred_raw))    

def train_svr():
    #C, gamma = tune(11)
    # previously found best values
    C = 1000.0
    gamma = 0.001
    
    X = Xall[0:m]
    models = []
    errors = []
    
    #train an SVR for all features
    for feature in range(np.shape(yall)[1]):
        y = yall[0:m, feature]
        ycv = yall[m+1:, feature]
        
        clf = svm.SVR(C=C, gamma=gamma)
        clf.fit(X, y)    
        pred = clf.predict(X)
        err = pred_error(y, pred)
    
        predcv = clf.predict(Xcv)
        errcv = pred_error(ycv, predcv)
        
        print [feature, err, errcv]    
        
        errors.append((err, errcv))
        models.append(clf)
        
    print errors
    #joblib.dump(clf, "%s/clf.pkl"%modeldir)
    
def train_forest():
    import random_forest
    
    errors = []
    feature = 1
    X = Xall[0:m,:]
    y = yall[0:m, feature]    
    ycv = yall[m+1:, feature]

    # train a random forest with different number of trees and plot error
    for trees in [1, 10, 20, 50, 100]:
        print "training forest %d" % trees
        forest = random_forest.create_forest(X.transpose(), y.reshape((1, -1)), trees)
        
        print "predict"
        pred = np.array([random_forest.classify_w_forest(forest, X.transpose()[:, i]) for i in range(m)])
        predcv = np.array([random_forest.classify_w_forest(forest, Xall.transpose()[:, i]) for i in range(m+1,3000)])
        #predcv = random_forest.classify_w_forest(forest, Xcv.transpose()).transpose()
        
        err = pred_error(y, pred)
        errcv = pred_error(ycv, predcv)
        print [trees, feature, err, errcv]
        errors.append((trees, feature, err, errcv))
   

# currently  the prediction errors computed above gave:
#[1, 1, 1.1310819619250145, 1.1069698773093029]
#[10, 1, 1.0451240743162091, 0.98967597112471728]
#[20, 1, 1.0293507462727491, 0.97791657855577008]
#[50, 1, 1.0260173044630487, 0.9723896953860699]
# [100, 1, 1.0256508773394162, 0.97134853426901291]

# to interpret these numbers consider the label scale that is normalized, more or less [-5, 5]

# in comparism SVR 
# [1, 0.0068497991726219872, 0.044224760442203054]

def train_sklearn_forest():
    
    errors = []
    feature = 1
    X = Xall[0:m,:]
    y = yall[0:m, feature]    
    ycv = yall[m+1:, feature]

    # train a random forest with different number of trees and plot error
    for trees in [1, 10, 20, 50]:
        print "training forest %d" % trees
        clf = RandomForestRegressor(n_estimators=trees, max_depth=20, )
        clf.fit(X, y)    
        pred = clf.predict(X)
        err = pred_error(y, pred)
        
        predcv = clf.predict(Xcv)
        errcv = pred_error(ycv, predcv)
        
        print [trees, feature, err, errcv]    
        
        errors.append((trees, feature, err, errcv))
        models.append(clf)


if __name__ == '__main__':
    train_forest()

