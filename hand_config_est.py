import sys
import glob
import os
import shutil
import numpy as np
import ipdb as pdb
import operator
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import preprocessing

#datadir = "ReducedFeatures/Features1x1_nod"
datadir = "NormalizedSample10krandom_reducedfeatures_1x1"
labeldir = "TrainData"
modeldir = "."
training_size = 0.66
model = "SVR"

try:
    datadir = sys.argv[1]
    labeldir = sys.argv[2]
    modeldir = sys.argv[3]
    model = sys.argv[4]
except IndexError:
    print "Usage: hand_config_est.py datadir labeldir modeldir model"
    print "Defaults used"

#ipca = joblib.load("%s/pca.pkl"%pcadir)

def load_data_simple(start, end):
    """ load data from files in datadir and labeldir, from start to end
    shuffle samples randomly    
    """
    data = []
    labels = []
    for i in np.random.permutation(range(start, end)): 
        data.append(np.loadtxt("%s/%d-0.txt" % (datadir,i), dtype=float))
        raw_label = np.loadtxt("%s/%d.txt" % (labeldir,i), dtype=float, )
        labels.append(raw_label.flatten())
    print "Data loaded (simple)"
    return np.array(data), np.array(labels)


def load_data(files, limit=0):
    """ load data from files in datadir and labeldir, from start to end
    shuffle samples randomly    
    """
    if not limit:
        limit = len(files)
    data = np.zeros((limit, 300))
    labels = np.zeros((limit, 60))
    files = np.random.permutation(files)
    for i, fname in enumerate([files[i] for i in range(limit)]): 
        data[i,:] = np.loadtxt("%s/%s" % (datadir, fname), dtype=float)
        numbers = fname.split('-')
        raw_label = np.loadtxt("%s/%s.txt" % (labeldir,numbers[0]), dtype=float, )
        labels[i,:] = raw_label.flatten()
    print "Data loaded"
    return np.array(data), np.array(labels)


# load training data to Xall and labels to yall, only if variables dont exists
#try:
#    Xall
#    yall
#except NameError:
#xRaw, yRaw = load_data_simple(0, 3000)
xRaw, yRaw = load_data(os.listdir(datadir))

# normalize both x and y
#preprocessing.scale(Xall, copy=False)

#scalerfunc=preprocessing.MinMaxScaler
scalerfunc=preprocessing.StandardScaler

scaler = scalerfunc().fit(xRaw)
Xall = scaler.transform(xRaw)
scaler = scalerfunc().fit(yRaw)
yall = scaler.transform(yRaw)

used = int(np.shape(xRaw)[0]*training_size)

# this is the cross validation set, from used to end
Xcv = Xall[used:, :]
#ycvRaw = yRaw[used:,1]
ycv = yall[used:, 1]

#plt.scatter(yall[:,1], yall[:,2])

m = used # number of training samples
C = 10.0 # C param for SVR
gamma = 0.1 # gamma param for gaussian kernel, inverse of variance
errors = []
models = []

def tune(features, train_size=used, cv_size=0, Cexp=1000.0, gammaexp=0.001):
    """
    trains SVR with different gamma and C param and chose the combinantion
    that results in lowest error on cross validation set
    """
    performance = {}
    if cv_size <= 0:
        cv_size = np.shape(Xall)[0]-train_size
    #vals = [0.1, 0.3, 1.0, 3.0, 10.0]
    vals = [0.1, 1.0, 10.0]
    for prog_i, feature in enumerate(features):
        for C in vals:
            C = C*Cexp
            for gamma in vals:
                gamma = gamma*gammaexp
                X = Xall[:train_size,:]
                y = yall[:train_size, feature]
                Xcv = Xall[train_size:train_size+cv_size,:]
                ycv = yall[train_size:train_size+cv_size, feature]
            
                clf = svm.SVR(C=C, gamma=gamma)
                clf.fit(X, y)
                
                pred = clf.predict(X)
                err = pred_error(y, pred)
            
                predcv = clf.predict(Xcv)
                errcv = pred_error(ycv, predcv)
            
                if (C, gamma) not in performance.keys():
                    performance[(C, gamma)] = []
                performance[(C, gamma)].append(errcv)
                
                print [feature, C, gamma, err, errcv]
    
        allopt = [min(performance.iteritems(), key=lambda x: x[1][i])[0] for i in range(prog_i+1)]
        print "optimal for each feature"
        for i in range(prog_i+1):
            print features[i], allopt[i], performance[allopt[i]][i]
        
        avg_performance = {key: np.mean(x) for key, x in performance.iteritems()}
        opt = min(avg_performance.iteritems(), key=operator.itemgetter(1))[0]
        opt_error = avg_performance[opt]
        
        print "optimal: ", opt, "error ", opt_error
    return allopt, opt

def pred_error(y, y_pred):
    #err = np.mean(abs((y-pred)/y))
    # average squared difference

    if scalerfunc == preprocessing.MinMaxScaler:
        y_raw = (y - scaler.min_[1]) / scaler.scale_[1]
        ypred_raw = (y_pred - scaler.min_[1]) / scaler.scale_[1]
    elif scalerfunc == preprocessing.StandardScaler:
        y_raw = y*scaler.std_[1] + scaler.mean_[1]
        ypred_raw = y_pred*scaler.std_[1] + scaler.mean_[1]
    else:
        assert False

    return np.mean(np.square(y_raw-ypred_raw))    

def train_svr():
    #C, gamma = tune(11)
    # previously found best values
    C = 10000.0
    gamma = 0.0001
    
    X = Xall[0:m]
    models = []
    errors = []
    
    #train an SVR for all features
    for feature in range(np.shape(yall)[1]):
        y = yall[0:m, feature]
        ycv = yall[m:, feature]
        
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
    
    for feature in range(np.shape(yall)[1]):    
        y = yall[0:m, feature] # RAW
        ycv = yall[m:, feature] # RAW
    
        # train a random forest with different number of trees and plot error
        for trees in [10]:
            print "training forest %d" % trees
            forest = random_forest.create_forest(X, y, trees,100,10)
            
            print "predict"
            pred = np.array([random_forest.classify_w_forest(forest, X[i,:]) for i in range(m)])
            predcv = np.array([random_forest.classify_w_forest(forest, Xall[m+i,:]) for i in range(len(ycv))])
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
    
    for feature in range(np.shape(yall)[1]):    
        y = yall[0:m, feature] # RAW
        ycv = yall[m:, feature] # RAW
    
        # train a random forest with different number of trees and plot error
        for trees in [20]:
            #print "training forest %d" % trees
            clf = RandomForestRegressor(n_estimators=trees)
            clf.fit(X, y)    
            pred = clf.predict(X)
            err = pred_error(y, pred)
            
            predcv = clf.predict(Xcv)
            errcv = pred_error(ycv, predcv)
            
            print [trees, feature, err, errcv]    
            
            errors.append((trees, feature, err, errcv))
            models.append(clf)


if __name__ == '__main__':
    res = tune(range(60), 6000, 2000)
    pickle.dump( res, open( "tune_result.p", "wb" ) )
    
    #train_sklearn_forest()
    #train_forest()
    
    #train_svr()

