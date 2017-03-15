"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import pandas
import scipy.misc as misc

'''
import data.face_data as fd
fd.load('~/data/Cropped_AMFEDPLUS.txt')
'''
def load(data_dir, subset='train'):

    label_df = pandas.read_csv(data_dir+'labels.txt',sep='\t')

    datax = np.zeros((32, 32, len(label_df), 3), dtype=np.float32)
    datay = np.zeros((len(label_df)), dtype=np.float32)
    X_data = [] 
    for index, row in label_df.iterrows():
        #print '~/data/'+'Cropped_AMFEDPLUS'+row[0]
        im = misc.imread(data_dir+row[0])
	    im2 = misc.imresize(im, [32,32])
        #datax = np.dstack((datax, im2))
        datax[:,:,index,0] = im2
        datax[:,:,index,1] = im2
        datax[:,:,index,2] = im2	
        #datay = np.append(datay, row[1])
        datay[index] = row[1]
        X_data.append (im2)
    datax = np.transpose(datax, (2, 3, 0, 1))

    
    if subset=='train':
        trainx = datax[:12000,:,:,:]
        trainy = datay[:12000]
        return trainx, np.array(trainy).astype(np.uint8)
    elif subset=='test':
        testx = datax[12001:,:,:,:]
        testy = datay[12001:]
        return testx, np.array(testy).astype(np.uint8)


    '''
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py','data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py','test_batch'))
        testx = test_data['x']
        testy = test_data['y']
	print testx
	print testx.shape
	print testy
	print testy.shape
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
    '''

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(os.path.join(data_dir), subset=subset)
        self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


