import sys
import glob
import os
#import warnings
import numpy as np
#import ipdb as pdb


def info_gain(S, Sl, Sr):
    """S is a the set of structure examples. Sl is the subset that
       goes to the left, Sr is the subset that goest to the right
       based on the split criteria at this node

       S - numpy array, columns are variables, rows are samples
    """
    #S is useless, S = Sl + Sr

    infgain = entropy(S) - ( np.shape(Sl)[0] / float(np.shape(S)[0]) * entropy(Sl) + np.shape(Sr)[0] / float(np.shape(S)[0]) * entropy(Sr))
    return infgain

def entropy(S):
    """ computes the entropy for S
    """
    #pdb.set_trace()
    #warnings.simplefilter("error", RuntimeWarning)
    det = np.linalg.det(covariant(S))
    if det==0:
        det = 1e-7
         
    #log = 0 if det == 0 else np.log(det)
    log = np.log(det)
    # Maybe the zero values causing some problems, try something big ;)
    #try:
    H = 0.5 * log + 0.5 * np.shape(S)[0] * (1 + 1.83787706641) # ln(2pi) = 1.8378..
    #except RuntimeWarning:
    #    print "Warning"
    return H

def covariant(S):
    if S.shape[0]==0:
        return np.zeros([1,1])
    
    if S.shape[0] == 1:
        return np.zeros([S.shape[1],S.shape[1]])
    
    covariance = np.matrix(np.cov(S,rowvar=0))
        
    return covariance