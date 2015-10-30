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

    infgain = entropy(S) - ( np.shape(Sl)[1] / float(np.shape(S)[1]) * entropy(Sl) + np.shape(Sr)[1] / float(np.shape(S)[1]) * entropy(Sr))
    return infgain

def entropy(S):
    """ computes the entropy for S
    """
    #pdb.set_trace()
    #warnings.simplefilter("error", RuntimeWarning)
    det = np.linalg.det(covariant(S))
    if det == 0:
        det = 1e-20
    #try:
    H = 0.5 * np.log(det) + 0.5 * np.shape(S)[1] * (1 + 1.83787706641) # ln(2pi) = 1.8378..
    #except RuntimeWarning:
    #    print "Warning"
    return H

def covariant(S):
    if S.size==0:
        covariance = np.zeros([1,1])
    elif S.size == 1:
        covariance = np.matrix(S)
    else:
        covariance = np.matrix(np.cov(S, rowvar=1)) 
        
    return covariance