import sys
import glob
import os
import shutil
import numpy as np
import ipdb as pdb
import operator
import matplotlib.pyplot as plt
import pickle
import argparse

#from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import preprocessing

#ipca = joblib.load("%s/pca.pkl"%pcadir)

#def load_data_simple(start, end):
#    """ load data from files in datadir and labeldir, from start to end
#    shuffle samples randomly
#    """
#    data = []
#    labels = []
#    for i in np.random.permutation(range(start, end)):
#        data.append(np.loadtxt("%s/%d-0.txt" % (datadir,i), dtype=float))
#        raw_label = np.loadtxt("%s/%d.txt" % (labeldir,i), dtype=float, )
#        labels.append(raw_label.flatten())
#    print "Data loaded (simple)"
#    return np.array(data), np.array(labels)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_data(datadir, labeldir, files, limit=0, delimiter='-', datatype='synt'):
    """ load data from files in datadir and labeldir, from start to end
    shuffle samples randomly
    """
    if not limit or limit > len(files):
        limit = len(files)
    data = np.zeros((limit, 300))
    labels = np.zeros((limit, 60))
    #files = np.random.permutation(files)
    for i, fname in enumerate([files[i] for i in range(limit)]):
        data[i, :] = np.loadtxt("%s/%s" % (datadir, fname), dtype=float)
        numbers = fname.split(delimiter)
        if datatype=='synt':
            raw_label = np.loadtxt("%s/%s.txt" % (labeldir, numbers[0]), dtype=float)
        elif datatype == 'real':
            raw_label = np.loadtxt("%s/%s.shand" % (labeldir, fname[:-4]), dtype=float, delimiter=',')
        else:
	    print "Loading file ", fname
            raw_label = np.zeros(60)
        labels[i, :] = raw_label.flatten()
    print "Data loaded"
    return np.array(data), np.array(labels)

# load training data to Xall and labels to yall, only if variables dont exists
#try:
#    Xall
#    yall
#except NameError:
#xRaw, yRaw = load_data_simple(0, 3000)


#plt.scatter(yall[:,1], yall[:,2])

#C = 10.0 # C param for SVR
#gamma = 0.1 # gamma param for gaussian kernel, inverse of variance
#errors = []
#models = []

def tune(XAlltr, XAllcv, yAlltr, yAllcv, features, Cexp=1000.0, gammaexp=0.0001):
    """
    trains SVR with different gamma and C param and chose the combinantion
    that results in lowest error on cross validation set
    """
    performance = {}
    #vals = [0.1, 0.3, 1.0, 3.0, 10.0]
    vals = [0.3, 0.6, 1.0, 3.0, 6.0]
    X = XAlltr
    Xcv = XAllcv
    for prog_i, feature in enumerate(features):
        for C in vals:
            C = C*Cexp
            for gamma in vals:
                gamma = gamma*gammaexp
                y = yAlltr[:, feature]
                ycv = yAllcv[:, feature]

                clf = svm.SVR(C=C, gamma=gamma)
                clf.fit(X, y)

                pred = clf.predict(X)
                err = pred_error(y, pred, feature)

                predcv = clf.predict(Xcv)
                errcv = pred_error(ycv, predcv, feature)

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

def pred_error(y, y_pred, feature):
    #err = np.mean(abs((y-pred)/y))
    # average squared difference

    if scalerfunc == preprocessing.MinMaxScaler:
        y_raw = (y - scaler.min_[feature]) / scaler.scale_[feature]
        ypred_raw = (y_pred - scaler.min_[feature]) / scaler.scale_[feature]
    elif scalerfunc == preprocessing.StandardScaler:
        y_raw = y*scaler.scale_[feature] + scaler.mean_[feature]
        ypred_raw = y_pred*scaler.scale_[feature] + scaler.mean_[feature]
    else:
        assert False

    return np.mean(np.square(y_raw-ypred_raw))


def train_svr(XAlltr, XAllcv, yAlltr, yAllcv, C=3000.0, gamma=0.0006):
    # previously found best values
    #C = 10000.0
    #gamma = 0.0001

    print "training SVR with C=%f gamma=%f" % (C, gamma)

    X = XAlltr
    Xcv = XAllcv
    models = []
    errors = []

    #train an SVR for all features
    for feature in range(np.shape(yAlltr)[1]):
        y = yAlltr[:, feature]
        ycv = yAllcv[:, feature]

        clf = svm.SVR(C=C, gamma=gamma)
        clf.fit(X, y)
        pred = clf.predict(X)
        err = pred_error(y, pred, feature)

        predcv = clf.predict(Xcv)
        errcv = pred_error(ycv, predcv, feature)

        print [feature, err, errcv]

        errors.append((err, errcv))
        models.append(clf)

    print errors

    return models, errors
    #joblib.dump(clf, "%s/clf.pkl"%modeldir)


def train_sklearn_forest(XAlltr, XAllcv, yAlltr, yAllcv, trees=20):
    errors = []
    models = []

    X = XAlltr
    Xcv = XAllcv

    print "training sklearn forset"

    for feature in range(np.shape(yAlltr)[1]):
        y = yAlltr[:, feature]
        ycv = yAllcv[:, feature]

        # train a random forest with different number of trees and plot error

        #print "training forest %d" % trees
        clf = RandomForestRegressor(n_estimators=trees, min_samples_leaf=30, max_depth=20)
        clf = RandomForestRegressor(n_estimators=trees)
        clf.fit(X, y)
        pred = clf.predict(X)
        err = pred_error(y, pred, feature)

        predcv = clf.predict(Xcv)
        errcv = pred_error(ycv, predcv, feature)

        print [trees, feature, err, errcv]

        errors.append((trees, feature, err, errcv))
        models.append(clf)

    return models, errors


def learning_curve(XAlltr, yAlltr, learn_func, step=100, steplimit=10000):
    results = {}
    m = step  
    indices = range(yAlltr.shape[0])    
    
    while True:
        if m > yAlltr.shape[0] or m > steplimit:
            break
        print "Datasize %d" % m
        np.random.shuffle(indices)
        models, errors = learn_func(XAlltr[indices[:m],:], yAlltr[indices[:m],:])
        
        results[m] = np.mean(errors, axis=0)
        print "Mean squared error", results[m]        
        
        m = m + step
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Second phase, estimates 60D hand configuration')

    parser.add_argument('method', type=str,
                        default='SVR',
                        help='Options: SVR, FOREST')

    parser.add_argument('--trainingdir', type=str, dest='trainingdir',
                        default='.',
                        help='Folder for training data')
    parser.add_argument('--testdir', type=str, dest='testdir',
                        default='.',
                        help='Folder for test data')
    parser.add_argument('--labeldir', type=str, dest='labeldir',
                        default='.',
                        help='Folder for label data')
    parser.add_argument('--output', type=str, dest='output',
                        default='',
                        help='File for text output')
    parser.add_argument('--deliiter', type=str, dest='delimiter',
                        default='-',
                        help='Delimiter to use for data file names')
    parser.add_argument('--traintype', type=str, dest='traintype',
                        default='synt',
                        help='synt or real')
    parser.add_argument('--testtype', type=str, dest='testtype',
                        default='synt',
                        help='synt or real')


    parser.add_argument('--cvrate', type=float, dest='cvrate',
                        default=1.0,
                        help='Ratio for training / test samples if they are in the same directory. Float [0,1]')
    parser.add_argument('--limit', type=int, dest='limit',
                        default=0,
                        help='Only use datalimit samples even if there is more training data')
    parser.add_argument('--step', type=int, dest='step',
                        default=100,
                        help='Datasize step for learning curve')
    parser.add_argument('--steplimit', type=int, dest='steplimit',
                        default=100,
                        help='Datasize limit for learning curve')                        

    args = parser.parse_args()

    if not args.output:
        import time
        args.output = time.strftime("%d-%m-%H-%M-%S") + "-%s.out" % (args.method)

    # load samples

    #filter file names that correspond to samples from sample range
    if args.cvrate != 1.0:
        assert args.traintype == 'synt' and args.testtype == 'synt'
        
        train_samples = np.array(range(3000))
        np.random.shuffle(train_samples)
        train_samples = train_samples[:int(3000*args.cvrate)]

        files_tr = [f for f in os.listdir(args.trainingdir)
                    if int(f.split(args.delimiter)[0]) in train_samples]
        files_cv = [f for f in os.listdir(args.testdir)
                    if int(f.split(args.delimiter)[0]) not in train_samples]
    else:
        files_tr = sorted(os.listdir(args.trainingdir))
        files_cv = sorted(os.listdir(args.testdir))

    if args.testtype == 'web':
	def comp_web_fnames(f1, f2):
	    f1_subj, f1_gest = f1[8:-4].split('_')
	    f2_subj, f2_gest = f2[8:-4].split('_')
	    if f1_subj != f2_subj:
		return cmp(int(f1_subj), int(f2_subj))
	    if is_number(f1_gest) and is_number(f2_gest):
		return cmp(int(f1_gest), int(f2_gest))
	    else:
		return cmp(f1_gest, f2_gest)
	files_cv = sorted(files_cv, comp_web_fnames)

    xRaw, yRaw = load_data(args.trainingdir, args.labeldir, files_tr, args.limit, args.delimiter, args.traintype)
    XAllcv, yAllcv = load_data(args.testdir, args.labeldir, files_cv, args.limit, args.delimiter, args.testtype)
    print "Train size: %d Test size: %d" % (xRaw.shape[0], XAllcv.shape[0])

    #scalerfunc=preprocessing.MinMaxScaler
    scalerfunc=preprocessing.StandardScaler
    scaler_x = scalerfunc().fit(xRaw)
    XAlltr = scaler_x.transform(xRaw)
    XAllcv = scaler_x.transform(XAllcv)

    scaler = scalerfunc().fit(yRaw)
    yAlltr = scaler.transform(yRaw)
    yAllcv = scaler.transform(yAllcv)
    
    models = []
    errors = []
    curve = {}
    if args.method == "SVR":
        models, errors = train_svr(XAlltr, XAllcv, yAlltr, yAllcv)
    elif args.method == "FOREST":
        models, errors = train_sklearn_forest(XAlltr, XAllcv, yAlltr, yAllcv)
    elif args.method == "SVRCURVE":
        curve = learning_curve(XAlltr, yAlltr, lambda xxx,yyyy: train_svr(xxx, XAllcv, yyyy, yAllcv), args.step, args.steplimit)
    elif args.method == "FORESTCURVE":
        curve = learning_curve(XAlltr, yAlltr, lambda xxx,yyy: train_sklearn_forest(xxx, XAllcv, yyy, yAllcv), args.step, args.steplimit)
    else:
        models, errors = tune(XAlltr, XAllcv, yAlltr, yAllcv, range(60))
    
    for k in sorted(curve.keys()):                                                                                              
        print k, curve[k][-2], curve[k][-1]    
    
    values = np.zeros((np.shape(XAllcv)[0], 60))
    for feature in range(60):
        values[:, feature] = models[feature].predict(XAllcv)*scaler.scale_[feature] + scaler.mean_[feature]
    #values = scaler.inverse_transform(values)    

    print errors
    print "average squared / nonsquared error"
    print np.mean(errors, axis=0)
    print np.sqrt(np.mean(errors, axis=0))
    print args.output
    pickle.dump((models, errors), open(args.output, "wb"))

    #res = tune(range(60), 6000, 2000)
    #pickle.dump( res, open( "tune_result.p", "wb" ) )

