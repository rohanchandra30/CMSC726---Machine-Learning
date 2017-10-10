import argparse
from collections import Counter, defaultdict
import random
import numpy as np
from sklearn.svm import SVC
from scipy import stats

from pylab import *

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=100,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code
    # Pos is 3's
    # Neg is 8's
    train_in_data = data.train_x[:args.limit]
    train_in_lab = data.train_y[:args.limit]
    test_in_data = data.test_x
    test_in_lab = data.test_y
    train_data = []
    train_labels = [] 
    test_data = []
    test_labels = [] 

    if args.limit > 0:
        for ii in range(args.limit):
            if(train_in_lab[ii]==3):
                train_data.append(train_in_data[ii])
                train_labels.append(3)
            elif (train_in_lab[ii]==8):
                train_data.append(train_in_data[ii])
                train_labels.append(8)
                
        for ii in range(len(test_in_data)):        
            if(test_in_lab[ii]==3):
                test_data.append(test_in_data[ii])
                test_labels.append(3)
            elif (test_in_lab[ii]==8):
                test_data.append(test_in_data[ii])
                test_labels.append(8)
                
                
    x = np.array(train_data)
    y = np.array(train_labels)
    print(y)
    x_test = np.array(test_data)
    y_test = np.array(test_labels)
    clf = SVC(C = 0.4,kernel = 'rbf')
    clf.fit(x,y)
    
    sup_inds = clf.support_
    sup_vec = x[sup_inds]
    
    
    print (clf.score(x_test,y_test))
    
    print (x.shape)
    #print sup_vec[2]
    #print test_labels 
    #print train_in_lab
    print (len(sup_vec))
    im_ar = np.reshape(sup_vec[260],(28,28))
    print (im_ar.shape)
    #imsave(im_ar,"sup_vec_0.png")
    imshow(im_ar, cmap="Greys_r")
    savefig('sup_vec_13.png')
    axis('off')
    show()        
            
    print("Done loading data")

