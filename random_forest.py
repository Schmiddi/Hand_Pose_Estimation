import sys
import glob
import os
import random
import numpy as np
import info_gain

def test_tree():
  n_samples = 100
  n_features = 27
  new_array=[]
  random.seed(1)
  for i in xrange(n_samples*n_features):
    new_array.append(random.random())

  new_array = np.array(new_array).reshape((n_features,n_samples))

  tree_node, estimation_node = create_tree(new_array)


def create_tree (D_n):
  random_index = random.sample(xrange(D_n.shape[1]), D_n.shape[1]/2)
  n=range(0,D_n.shape[1])
  U_n=D_n[ :,random_index]
  E_n=D_n[ :,set(n)-set(random_index)]
  tree_node=[]
  estimation_node=[]
  max_h = np.log2(D_n.shape[1])
  return create_node(U_n, E_n, 0, max_h, tree_node, estimation_node)

def create_node(U_n,E_n,h,max_h,tree_node,estimation_node):
  T=np.arange(0,1,0.1)
  p=0.4
  s=1+np.random.binomial(U_n.shape[0],p)
  m_0=100
  k_n=10
  s_index=random.sample(xrange(U_n.shape[0]),s)
  mu=min(m_0,U_n.shape[1])
  max_t_index = []
  max_value = []
  for i in s_index:
    mu_index=random.sample(xrange(U_n.shape[1]),mu)
    gains = []
    for t in T:
      U_1 = U_n[:, U_n[i,:] < t]
      U_2 = U_n[:, U_n[i,:] >= t]
      gains.append(info_gain.info_gain(U_n, U_1, U_2))

    index = np.argmax(np.array(gains))
    max_t_index.append(index)
    max_value.append(gains[index])

  index = np.argmax(np.array(max_value))
  max_feature = s_index[index]
  max_t = max_t_index[index]    

  U_1 = U_n[:, U_n[max_feature,:] < max_t]
  U_2 = U_n[:, U_n[max_feature,:] >= max_t]
  E_1 = E_n[:, E_n[max_feature,:] < max_t]
  E_2 = E_n[:, E_n[max_feature,:] >= max_t]
  h+=1
  
  if E_1.shape[1]<k_n or E_2.shape[1]<k_n or h>max_h:
    # Create new end node
    estimation_node.append(E_n)
    node = [None, None, len(estimation_node)-1, None]
    tree_node.append(node)
    return tree_node, estimation_node


  # create node
  index_node = len(tree_node)
  node = [max_feature, max_t, index_node+1, None]
  tree_node.append(node)
  tree_node, estimation_node = create_node(U_1, E_1, h, max_h, tree_node, estimation_node)
  tree_node[index_node][3] = len(tree_node)
  tree_node, estimation_node = create_node(U_2, E_2, h, max_h, tree_node, estimation_node)

  return tree_node, estimation_node

