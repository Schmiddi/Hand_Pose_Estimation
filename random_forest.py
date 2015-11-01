import random
import numpy as np
import info_gain

def test_tree():
    n_samples = 5000
    n_features = 27
    new_array = []
    random.seed(1)
    n_labels = np.zeros([n_samples,1])
    new_array = np.zeros([n_samples, n_features])
    
    numLabels = 3
    
    for i in xrange(n_samples):
        n_labels[i,0] = random.randint(1,numLabels) #1,2,3
        for j in xrange(n_features):
            new_array[i,j] = np.random.uniform((n_labels[i,0]-1)*1/numLabels,n_labels[i,0]*1/numLabels)
        

    forest = create_forest(new_array, n_labels,1)
    #print forest
    for i in xrange(20):
        print classify_w_forest(forest, new_array[i,:]),n_labels[i,0]
    print "hello world"

'''
    D_n: rows = samples, col = features (Feature vector: x = [0,1]^d)
    D_n_labels: row=samples, col = labels (Label vector: y =  R^q)
    num_trees: number of trees to generate (default: 10)
    steps: step size for the thresholds (default: 100)
    m_0: amount of structure data we maximal should use for the next construction step (default: 1e7)
    k_n: minimum amount of estimations examples per node (default: 30)
    p: predefined probability for feature selection (default: 0.2)
'''
def create_forest(D_n, D_n_labels, num_trees=10, steps = 100, k_n=30, m_0=1e7, p=0.2):
    forest = []
    for i in xrange(num_trees):
        print "Creating tree %d\r"%i
        forest.append(create_tree (D_n, D_n_labels, steps, k_n, m_0, p))
    
    return forest

def classify_w_forest(forest, sample):
    prediction = 0
    for tree in forest:
        prediction = prediction + classify_w_tree(tree,sample)
    
    return prediction / len(forest)

# sample:  Feature vector: x = [0,1]^d
def classify_w_tree(tree, sample):
    tree_node, estimation_node = tree
    
    i = 0
    while True:
        if tree_node[i][0] is None:
            # Yeah it's a leave node, we can finally predict!!
            e_labels = estimation_node[tree_node[i][1]]
            # Average each col of the estimation data
            prediction = np.sum(e_labels, axis=0)/e_labels.shape[0]
            return prediction
        
        # else - non leave node
        if sample[tree_node[i][0]] < tree_node[i][1]:
            i = tree_node[i][2]
        else:
            i = tree_node[i][3]
        
    
'''
    D_n: rows = samples, col = features (Feature vector: x = [0,1]^d)
    D_n_labels: row=samples, col = labels (Label vector: y =  R^q)
    steps: step size for the thresholds (default: 100)
    m_0: amount of structure data we maximal should use for the next construction step (default: 1e7)
    k_n: minimum amount of estimations examples per node (default: 30)
    p: predefined probability for feature selection (default: 0.2)
'''
def create_tree (D_n, D_n_labels, steps = 100, k_n=30, m_0=1e7, p=0.2):
    # Split the data set D_n into two parts
    # Put floor(n/2) random samples into U_n,
    # and the remaining ceil(n/2) samples in E_n 
    random_index = random.sample(xrange(D_n.shape[0]), D_n.shape[0] / 2)
    n = range(0, D_n.shape[0])
    ## Structure data
    U_n = D_n[random_index, :]
    if(len(D_n_labels.shape) == 1):
        D_n_labels = D_n_labels.reshape(D_n_labels.shape[0],1)
        
    U_n_labels = D_n_labels[random_index, :]
    ## Estimation data
    E_n = D_n[ list(set(n) - set(random_index)),:]
    E_n_labels = D_n_labels[ list(set(n) - set(random_index)),:]
    tree_node = []
    estimation_node = []
    # Set the maximal height of a tree
    max_h = np.log2(D_n.shape[0])
    return create_node(U_n, U_n_labels, E_n, E_n_labels, 0, max_h, tree_node, estimation_node, steps, k_n, m_0, p)


'''
    steps: step size for the thresholds (default: 100)
    m_0: amount of structure data we maximal should use for the next construction step (default: 1e7)
    k_n: minimum amount of estimations examples per node (default: 30)
    p: predefined probability for feature selection (default: 0.2)
'''
def create_node(U_n, U_n_labels, E_n, E_n_labels, h, max_h, tree_node, estimation_node, steps = 100, k_n=30, m_0=1e7, p=0.2):
    # Define tuning parameters
    ## Define candidate thresholds from 0 to 1 in 0.1 steps
    #T = np.arange(0, 1, steps)
    ## Number of distinct features we select for tree construction
    s = 1 + np.random.binomial(U_n.shape[1], p)
    ## Get s random indices
    s_index = random.sample(xrange(U_n.shape[1]), s)
    ## Calculate the number of structure data to use for the next construction step
    mu = min(m_0, U_n.shape[0])
    max_t_index = []
    max_value = []
    # Iterate of all feature indices
    for fi in s_index:
        # Get random indices of the structure data (for each feature different data)
        mu_index = random.sample(xrange(U_n.shape[0]), mu)
        gains = []
        tmin = np.min(U_n[:,fi])
        tmax = np.max(U_n[:,fi])
        trange = tmax-tmin
        T = np.arange(tmin,tmax,trange/steps)
        for t in T:
            # Split the M structure data samples based on the threshold t
            U_1_labels = U_n_labels[U_n[mu_index,fi] < t,:]
            U_2_labels = U_n_labels[U_n[mu_index, fi] >= t, :]
            # Calculate the information gain using this split
            if U_1_labels.size == 0 or U_2_labels.size ==0:
                gains.append(0)
            else:
                gains.append(info_gain.info_gain(U_n_labels[mu_index,:], U_1_labels, U_2_labels))
        
        # Get the index of the threshold maximizing the info gain, and the info gain value 
        index = np.argmax(np.array(gains))
        max_t_index.append(T[index])
        max_value.append(gains[index])
    
    # Get the feature and threshold which maximize the info gain
    index = np.argmax(np.array(max_value))
    max_feature = s_index[index]
    max_t = max_t_index[index]

    # Split the structure and estimation data based on the best feature and threshold
    U_1 = U_n[U_n[:, max_feature] < max_t, :]
    U_1_labels = U_n_labels[U_n[:, max_feature] < max_t,:]
    U_2 = U_n[U_n[:, max_feature] >= max_t, :]
    U_2_labels = U_n_labels[U_n[:, max_feature] >= max_t,:]
    E_1 = E_n[E_n[:, max_feature] < max_t, :]
    E_1_labels = E_n_labels[E_n[:, max_feature] < max_t,:]
    E_2 = E_n[E_n[:, max_feature] >= max_t, :]
    E_2_labels = E_n_labels[E_n[:, max_feature] >= max_t,:]
    h += 1
    
    # Check is after the split the tree is still valid
    if E_1.shape[0] < k_n or E_2.shape[0] < k_n or h > max_h:
        # Create new end node
        estimation_node.append(E_n_labels)
        # If it is a leave node, we place the index of the estimation node array holding the estimation nodes of this leave in the second field, the rest None
        node = [None, len(estimation_node) - 1, None, None]
        tree_node.append(node)
        return tree_node, estimation_node


    # If it is valid, create the new nodes
    index_node = len(tree_node)
    # Node [ best feature, best threshold, index left node, index right node]
    node = [max_feature, max_t, index_node + 1, None]
    tree_node.append(node)
    tree_node, estimation_node = create_node(U_1, U_1_labels, E_1, E_1_labels, h, max_h, tree_node, estimation_node, steps, k_n, m_0, p)
    tree_node[index_node][3] = len(tree_node)
    tree_node, estimation_node = create_node(U_2, U_2_labels, E_2, E_2_labels, h, max_h, tree_node, estimation_node, steps, k_n, m_0, p)

    return tree_node, estimation_node

if __name__ == '__main__':
    test_tree()
