# -*- coding: utf-8 -*-
# This implementation aims at unifying the implementation of the 2-MLP architecture.
# Instead of calling 2 MLP objects, there is now 1 object with a unified Theano graph.
# 

import cPickle
import os
import sys
import time
import datetime

import numpy
import pylab
import math

import theano
import theano.tensor as T

import scipy.io.wavfile as wv
    
    
def load_npz_data(dataset):
    ''' Loads the dataset from a npz archive
    Does the same thing as load_data() but with an additional dataset
    :type dataset: string
    '''

    print '... loading data from '+dataset

    data = numpy.load(dataset)

    def shared_dataset_WNN(data_x, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        return shared_x
        
    def shared_dataset_NSNN(data_x, data_y, borrow=True):
        data_x /= 560
        data_y /= 560
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'float64')

    train_set_x_WNN = shared_dataset_WNN(data['train_WNN'])
    valid_set_x_WNN = shared_dataset_WNN(data['valid_WNN'])
    test_set_x_WNN = shared_dataset_WNN(data['test_WNN'])
    sentence_x_WNN = shared_dataset_WNN(data['sentence_WNN'])
    
    train_set_x, train_set_y = shared_dataset_NSNN(data['train_in_NSNN'],data['train_out_NSNN'])
    valid_set_x, valid_set_y = shared_dataset_NSNN(data['valid_in_NSNN'],data['valid_out_NSNN'])
    test_set_x, test_set_y = shared_dataset_NSNN(data['test_in_NSNN'],data['test_out_NSNN'])
    sentence_x, sentence_y = shared_dataset_NSNN(data['sentence_in_NSNN'],data['sentence_out_NSNN'])

    rval1 = [(train_set_x_WNN), (valid_set_x_WNN),
            (test_set_x_WNN), (sentence_x_WNN)]
    rval2 = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), (sentence_x, sentence_y)]
    return rval1, rval2
    
    
    

class HiddenLayer(object):
    def __init__(self, rng, layerInput, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        NOTE : The nonlinearity used here is tanh
        
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type layerInput: theano.tensor.dmatrix
        :param layerInput: a symbolic tensor of shape (n_examples, n_in)
        
        :type n_in: int
        :param n_in: dimensionality of layerInput
        
        :type n_out: int
        :param n_out: number of hidden units
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = layerInput

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        # activation function used (among other things).
        # For example, results presented in [Xavier10] suggest that you
        # should use 4 times larger initial weights for sigmoid
        # compared to tanh
        # We have no info for other function, so we use the same as
        # tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX) #+ 10 # for ReLU
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(layerInput, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # self.output = T.max(0,lin_output) # ReLU
                       
        # parameters of the model
        self.params = [self.W, self.b]


class OutputLinear(object):
    def __init__(self, layerInput, n_in, n_out, W=None, b=None):
        
        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            W = theano.shared(value=numpy.zeros((n_in, n_out),
                                   dtype=theano.config.floatX),
                                   name='W', borrow=True)
        if b is None:
            # initialize the baises b as a vector of n_out 0s                           
            b = theano.shared(value=numpy.zeros((n_out,),
                                   dtype=theano.config.floatX),
                                   name='b', borrow=True)
                                   
        self.W = W
        self.b = b

        # compute vector of real values in symbolic form
        self.y_pred = T.dot(layerInput, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

    def errors(self, y):
        
        #print 'computing the error...'

        # check if y has same dimension of y_pred
#        if y.ndim != self.y_pred.ndim:
#            raise TypeError('y should be the same shape as self.y_pred',
#                            ('y', y.type, 'y_pred', self.y_pred.type))
#        else:
#            return T.mean((self.y_pred - y)**2)
            
        return T.mean((self.y_pred - y)**2)


class MLP(object):
    """Multi-Layer Perceptron Class.
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    """

    def __init__(self, rng, layerInput_NSNN, layerInput_WNN, 
                 n_in_NSNN, n_hidden_NSNN, n_out_NSNN, 
                 n_in_WNN, n_hidden_WNN, n_out_WNN):
        """Initialize the parameters for the multilayer perceptron.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type layerInput_NSNN: theano.tensor.TensorType
        :param layerInput_NSNN: symbolic variable that describes the input of the
        Next Sample Neural Network (one minibatch)
        
        :type layerInput_WNN: theano.tensor.TensorType
        :param layerInput_WNN: symbolic variable that describes the input of the
        Weights Neural Network (one minibatch)
        
        :type n_in: int
        :param n_in: number of layerInput units, the dimension of the space in
        which the datapoints lie
        
        :type n_hidden: list of int
        :param n_hidden: the number of entries in the list corresponds to the desired number of hidden layers;
                         each entry in the list is the number of hidden units for each of the hidden layers
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        """
        
        nb_biases_NSNN = sum(n_hidden_NSNN) + n_out_NSNN
                                  
        self.hiddenLayerWNN = HiddenLayer(rng=rng, layerInput=layerInput_WNN,
                                       n_in=n_in_WNN, n_out=n_hidden_WNN,
                                       activation=T.tanh)

        self.outputLayerWNN = OutputLinear(layerInput=self.hiddenLayerWNN.output,
                                        n_in=n_hidden_WNN,
                                        n_out=nb_biases_NSNN)
                                        
        # First hidden layer of NSNN                            
        self.hiddenLayer1NSNN = HiddenLayer(rng=rng, layerInput=layerInput_NSNN,
                                       n_in=n_in_NSNN, n_out=n_hidden_NSNN[0],
                                       b=self.outputLayerWNN.y_pred[0,0:n_hidden_NSNN[0]],
                                       activation=T.tanh)
        
        # Second hidden layer of NSNN                               
        self.hiddenLayer2NSNN = HiddenLayer(rng=rng, layerInput=self.hiddenLayer1NSNN.output,
                                       n_in=n_hidden_NSNN[0], n_out=n_hidden_NSNN[1],
                                       b=self.outputLayerWNN.y_pred[0,n_hidden_NSNN[0]:n_hidden_NSNN[0]+n_hidden_NSNN[1]],
                                       activation=T.tanh)
        # Output layer of NSNN   
        self.outputLayerNSNN = OutputLinear(layerInput=self.hiddenLayer2NSNN.output,
                                        n_in=n_hidden_NSNN[1],
                                        b=self.outputLayerWNN.y_pred[0,n_hidden_NSNN[0]+n_hidden_NSNN[1]:],
                                        n_out=n_out_NSNN)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayerNSNN.W).sum() + abs(self.outputLayerNSNN.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayerNSNN.W ** 2).sum() + (self.outputLayerNSNN.W ** 2).sum()

        # computing the mean square errors
        self.errors = self.outputLayerNSNN.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        # self.params = self.hiddenLayerNSNN.params + self.outputLayerNSNN.params + self.hiddenLayerWNN.params + self.outputLayerWNN.params
        self.params = [self.hiddenLayerNSNN.W] + [self.outputLayerNSNN.W] + self.hiddenLayerWNN.params + self.outputLayerWNN.params

    def saveParams(self, dataPath, MLPtype):
        """ Pickle the W1, b1, W2 and b2 parameters of the trained network
        input
            datapath: string (path to the folder with data)
            MLPtype: string (WNN or NSNN)
        """
        
        filename = 'ARCH2_'+ MLPtype +'_params.pkl' # ARCH2_WNN_params.pkl"
        f = file(dataPath+filename, 'wb')
        cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        self.showWeights()
        
    def loadParams(self, dataPath, MLPtype):
        """ Load the W1, b1, W2 and b2 parameters of the trained network
        input
            datapath: string (path to the folder with data)
            MLPtype: string (WNN or NSNN)
        """
        
        filename = 'ARCH2_'+ MLPtype +'_params.pkl' # ARCH2_WNN_params.pkl"
        f = file(dataPath+filename, 'rb')

#        # Check if the parameters are of the expected size
#        loadedParams = cPickle.load(f)
#        if loadedParams.shape == self.params.shape:
#            self.params = loadedParams
#        else:
#            raise TypeError('Imported parameters are not of the expected shape',
#                            ('loadedParams', loadedParams.type, 'self.params', self.params.type))
            
        self.params = cPickle.load(f)
        f.close()
        #self.showWeights()
        
    def showWeights(self):
        """This function should give a sense of the current weights of the MLP, either by printing or plotting"""
        print 'W1_NSNN: ' + str(self.params[0].get_value().shape)
        print self.params[0].get_value()
        print 'W2_NSNN: ' + str(self.params[1].get_value().shape)
        print self.params[1].get_value()
        print 'W1_WNN: ' + str(self.params[2].get_value().shape)
        print self.params[2].get_value()
        print 'b1_WNN: ' + str(self.params[3].get_value().shape)
        print self.params[3].get_value()
        print 'W2_WNN: ' + str(self.params[4].get_value().shape)
        print self.params[4].get_value()
        print 'b2_WNN: ' + str(self.params[5].get_value().shape)
        print self.params[5].get_value()
        
        #pylab.figure()
        #pylab.imshow(self.params[0].get_value())
        #pylab.title('W1')
        
        

def test_mlp(learning_rate=[0.15,0.15], L1_reg=[0.0,0.0], L2_reg=[0.000001,0.000001], n_epochs=1000,
             dataPath='', savePath='', fileNameData='', batch_size=20, n_hidden=[5000,10], n_out=[2441,1]):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float [WNN,NSNN]
    :param learning_rate: learning rate used (factor for the stochastic
    gradient
    
    :type L1_reg: float [WNN,NSNN]
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)
    
    :type L2_reg: float [WNN,NSNN]
    :param L2_reg: L2-norm's weight when added to the cost (seeself.params[0].get_value().shape
    regularization)
    
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    
    :type n_hidden: int [WNN,NSNN]
    :param n_hidden: number of hidden units
    
    """
    
    # Check if the data is already in memory
    if 'datasetWNN' in locals():   
        print('...data was already loaded')
    else:
        datasetWNN, datasetNSNN = load_npz_data(dataPath+fileNameData)
        
        # Load train/valid/test sets for WNN
        train_set_x_WNN = datasetWNN[0]
        valid_set_x_WNN = datasetWNN[1]
        test_set_x_WNN = datasetWNN[2]
        sentence_x_WNN = datasetWNN[3]
        
        # Load train/valid/test sets for NSNN
        train_set_x_NSNN, train_set_y_NSNN = datasetNSNN[0]
        valid_set_x_NSNN, valid_set_y_NSNN = datasetNSNN[1]
        test_set_x_NSNN, test_set_y_NSNN = datasetNSNN[2]
        sentence_x_NSNN, sentence_y_NSNN = datasetNSNN[3]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x_WNN.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x_WNN.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x_WNN.get_value(borrow=True).shape[0] / batch_size
        n_sentence_samples = sentence_x_WNN.get_value(borrow=True).shape[0]
    
    
    ######################
    # BUILD ACTUAL MODEL # dataset
    ######################
    print '... building the model'
    
    rng = numpy.random.RandomState(1234)
    
    # Define the 2nd architecture
    nb_hidden_units_NSNN = n_hidden[1]
    nb_out_NSNN = 1
    nb_in_NSNN = train_set_x_NSNN.get_value().shape[1]
    print "NSNN..."
    print "     W1: "+str(nb_hidden_units_NSNN[0])+" x "+str(nb_in_NSNN)    
    print "     b1: "+str(nb_hidden_units_NSNN[0])+" x 1"
    print "     W2: "+str(nb_hidden_units_NSNN[1])+" x "+str(nb_hidden_units_NSNN[0])    
    print "     b2: "+str(nb_hidden_units_NSNN[1])+" x 1"
    print "     W3: "+str(nb_out_NSNN)+" x "+str(nb_hidden_units_NSNN[1])    
    print "     b3: "+str(nb_out_NSNN)+" x 1"
    
    nb_in_WNN = train_set_x_WNN.get_value().shape[1]
    nb_out_WNN = nb_hidden_units_NSNN + nb_out_NSNN
    nb_hidden_units_WNN = n_hidden[0]
    print "WNN..."
    print "     W1: "+str(nb_hidden_units_WNN)+" x "+str(nb_in_WNN)    
    print "     b1: "+str(nb_hidden_units_WNN)+" x 1"
    print "     W2: "+str(nb_out_WNN)+" x "+str(nb_hidden_units_WNN)    
    print "     b2: "+str(nb_out_WNN)+" x 1"
    
    # allocate symbolic variables for the WNN data
    x_WNN = T.matrix('x_WNN') # MLP input
    y_WNN = T.vector('y_WNN') # MLP output
    index = T.lscalar() # index to a minibatch
    
    # allocate symbolic variables for the NSNN data
    x_NSNN = T.matrix('x_NSNN')
    y_NSNN = T.dmatrix('y_NSNN')
    previous_samples = T.matrix('previous_samples_NSNN')
    
    # construct the MLP class
    NN = MLP(rng=rng, layerInput_NSNN=x_NSNN, layerInput_WNN=x_WNN, 
                 n_in_NSNN=nb_in_NSNN, n_hidden_NSNN=nb_hidden_units_NSNN, n_out_NSNN=nb_out_NSNN, 
                 n_in_WNN=nb_in_WNN, n_hidden_WNN=nb_hidden_units_WNN, n_out_WNN=nb_out_WNN)
                     
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost_NN = NN.errors(y_NSNN) \
         + L1_reg[1] * NN.L1 \
         + L2_reg[1] * NN.L2_sqr             
                
                
    # NEURAL NETWORK #############################################
                
    # For debugging, a function that outputs the y_pred of WNN
    fprop_WNN = theano.function(inputs=[index],
            outputs=NN.outputLayerWNN.y_pred,
            givens={x_WNN: train_set_x_WNN[index * batch_size:(index + 1) * batch_size]})

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model_NN = theano.function(inputs=[index],
            outputs=NN.errors(y_NSNN),
            givens={              
                x_NSNN: test_set_x_NSNN[index * batch_size:(index + 1) * batch_size],
                y_NSNN: test_set_y_NSNN[index * batch_size:(index + 1) * batch_size],
                x_WNN: test_set_x_WNN[index * batch_size:(index + 1) * batch_size]})

    validate_model_NN = theano.function(inputs=[index],
            outputs=NN.errors(y_NSNN),
            givens={
                x_NSNN: valid_set_x_NSNN[index * batch_size:(index + 1) * batch_size],
                y_NSNN: valid_set_y_NSNN[index * batch_size:(index + 1) * batch_size],
                x_WNN: valid_set_x_WNN[index * batch_size:(index + 1) * batch_size]})

    # compiling a Theano function that reconstructs a sentence
    yrec_model_NN = theano.function(inputs=[index],
            outputs=NN.outputLayerNSNN.y_pred,
            givens={
                x_NSNN: sentence_x_NSNN[index:index+1],
                x_WNN: sentence_x_WNN[index:index+1]})

    # compiling a Theano function that generates the next sample
    ygen_model_NN = theano.function(inputs=[index,previous_samples],
            outputs=NN.outputLayerNSNN.y_pred,
            givens={
                x_NSNN: previous_samples,
                x_WNN: sentence_x_WNN[index:index+1]})

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams_NN = []
    for param in NN.params:
        gparam = T.grad(cost_NN, param)
        gparams_NN.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates_NN = []
    for param, gparam in zip(NN.params, gparams_NN):
        updates_NN.append((param, param - learning_rate[1] * gparam))
                        
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model_NN = theano.function(inputs=[index], outputs=cost_NN,
            updates=updates_NN,
            givens={
                x_NSNN: train_set_x_NSNN[index * batch_size:(index + 1) * batch_size],
                y_NSNN: train_set_y_NSNN[index * batch_size:(index + 1) * batch_size],
                x_WNN: train_set_x_WNN[index * batch_size:(index + 1) * batch_size]})
                
    ##########################################################################

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    # Create a folder and a log to record what's happening
    date_format = '%Y%m%d%H%M%S'
    string_now = datetime.datetime.now().strftime(date_format)
    folderName = 'arch2uni_'+string_now+'/'
    if not os.path.exists(savePath+folderName):
        os.makedirs(savePath+folderName)
    savePath = savePath+folderName
    log_name = 'arch2uni_log_file_'+string_now+'.txt'
    log_file = open(savePath+log_name, 'w')
    log_file.write(str(datetime.datetime.now())+'\n')
    
     
    
    # Write the hyperparameters of the model
    log_file.write('Second architecture UNIFIED - NSNN & WNN\n')
    log_file.write('--------------------------------\n')
    log_file.write('WNN:\n')
    log_file.write('    Nb of input units: '+str(nb_in_WNN)+'\n')
    log_file.write('    Nb of hidden units: '+str(nb_hidden_units_WNN)+'\n')
    log_file.write('    Nb of output units: '+str(nb_out_WNN)+'\n')
    log_file.write('NSNN:\n')
    log_file.write('    Nb of input units: '+str(nb_in_NSNN)+'\n')
    log_file.write('    Nb of hidden units (1): '+str(nb_hidden_units_NSNN[0])+'\n')
    log_file.write('    Nb of hidden units (2): '+str(nb_hidden_units_NSNN[1])+'\n')
    log_file.write('    Nb of output units: '+str(nb_out_NSNN)+'\n')
    
    # Hyperparameter values
    log_file.write('Hyperparameters (WNN, NSNN): \n')
    log_file.write('    Learning rate: '+str(learning_rate)+'\n')
    log_file.write('    L1 weight decay: '+str(L1_reg)+'\n')
    log_file.write('    L2 weight decay: '+str(L2_reg)+'\n')
    log_file.write('    Batch size: '+str(batch_size)+'\n')
    log_file.write('    Number of epochs: '+str(n_epochs)+'\n')
    
    # Used data
    log_file.write('\n')
    log_file.write('Data: \n')
    log_file.write('    File: '+savePath+fileNameData+'\n')
    log_file.write('    Number of training examples: '+str(train_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of validation examples: '+str(valid_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of test examples: '+str(test_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of reconstruction/generation examples: '+str(sentence_y_NSNN.get_value().shape)+'\n')

    best_params = None
    this_validation_loss = numpy.inf
    best_validation_loss = numpy.inf
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    train_err = []
    train_losses = numpy.zeros(n_train_batches)
    validation_losses = numpy.zeros(n_valid_batches)
    test_losses = numpy.zeros(n_test_batches)
    valid_err = []
    
    y_pred = numpy.zeros(n_sentence_samples)
    
    std_factor = [0.1, 1, 2, 5,] #[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

    epoch = 0
    done_looping = False

    log_file.write('\nTraining\n')
    
    while (epoch < n_epochs) and not done_looping:
        epoch = epoch + 1
        print('Epoch '+str(epoch)+':')
        
        # Training set
        for i in xrange(n_train_batches): # xrange(10000): #
            train_losses[i] = (560**2)*train_model_NN(i)
            #NSNN.showWeights()
            #WNN.showWeights()
            #raw_input("PRESS ENTER TO CONTINUE.")
            if i%10000 == 0:
                print('    Training iteration '+str(i)+'/'+str(n_train_batches))
                if math.isnan(train_losses[i]):
                    print('Training diverged at epoch '+str(epoch))
                    log_file.write('\n\nTraining diverged at epoch '+str(epoch)+', before iteration '+str(i)+'. Aborting training. \n')
                    done_looping = True
                    break
                    # log_file.close()
                    # raise Exception("Training diverged")
        this_train_loss = numpy.mean(train_losses)
        train_err.append(this_train_loss)
    
        # Validation set
        if not done_looping:
            for i in xrange(n_valid_batches): #xrange(100): # 
                if i%10000 == 0:
                    print('    Validation iteration '+str(i)+'/'+str(n_valid_batches))
                validation_losses[i] = (560**2)*validate_model_NN(i)
            this_validation_loss = numpy.mean(validation_losses)        
            valid_err.append(this_validation_loss)
            
            print('epoch %i, train error %f, validation error %f' %
                 (epoch, this_train_loss, this_validation_loss))
            log_file.write('Epoch %i, train error %f, validation error %f' %
                 (epoch, this_train_loss, this_validation_loss))
            log_file.write('\n')
#        else:
#            print('Training diverged at epoch '+str(epoch))
#            log_file.write('\n\nTraining diverged at epoch '+str(epoch)+'. Aborting training. \n')
#            # log_file.close()
#            # raise Exception("Training diverged")

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_epoch = epoch
        
            # Save the parameters of the model
            NN.saveParams(savePath, 'NN')
            
            # Generate the sentence
            if epoch%15 == 0:
                print '\n... ... Generating'
                for ind,k in enumerate(std_factor):
                    y_gen = numpy.zeros(n_sentence_samples)
                    presamples = numpy.zeros(240) #sentence_x_NSNN.get_value()[2500]
                    for i in xrange(n_sentence_samples): #xrange(1000): #
                        y_gen[i] = numpy.random.normal(ygen_model_NN(i,presamples.reshape((1, 240))),
                                                       k*numpy.sqrt(min(train_err)/(560**2)))
                        presamples = numpy.roll(presamples, -1)
                        presamples[-1] = y_gen[i]
                        
                    output = numpy.int16(y_gen*560)
                    wv.write(savePath+'Train_generated_data_Epoch'+str(epoch)+'_factor'+str(ind)+'.wav', 16000, output)
            
#        raw_input("PRESS ENTER TO CONTINUE.")
            
        # Check if the training has improved the validation error; if not, stop training
        if epoch > 20:
            if numpy.mean(valid_err[-3:]) > numpy.mean(valid_err[-6:-3]):
                done_looping = True
                break

    # Load the best model
    try:
        NN.loadParams(savePath, 'NN')

        for i in xrange(n_test_batches):
            if i%10000 == 0:
                print('    Testing iteration '+str(i)+'/'+str(n_test_batches))
            test_losses[i] = (560**2)*test_model_NN(i)
        test_score = numpy.mean(test_losses)
              
        print(('\n Optimization complete. Best validation score of %f '
               'obtained at epoch %i, with test performance %f') %
              (best_validation_loss, best_epoch, test_score))
        log_file.write(('\nOptimization complete. Best validation score of %f '
               'obtained at epoch %i, with test performance %f \n') %
              (best_validation_loss, best_epoch, test_score))
        
        # Plot the training graph                      
        pylab.figure()
        pylab.plot(range(epoch), train_err)
        pylab.plot(range(epoch), valid_err)
        pylab.xlabel('epoch')
        pylab.ylabel('MSE')
        pylab.legend(['train', 'valid'])
        pylab.savefig(savePath+'error.png', format='png')          
    
        # Reconstruct the sentence
        print '... ... reconstructing'
        for i in xrange(n_sentence_samples): #xrange(1000): #
            if i%10000 == 0:
                print('    Reconstruction iteration '+str(i)+'/'+str(n_sentence_samples))                     
            y_pred[i] = yrec_model_NN(i)
        
        # Save in wav format and save a figure
        reconstructed_output = numpy.int16(y_pred*560)
        wv.write(savePath+'predicted_data.wav', 16000, reconstructed_output)
        
        original_output = numpy.int16(sentence_y_NSNN.get_value()*560)
        wv.write(savePath+'original_data.wav', 16000, original_output)
        
        pylab.figure()
        pylab.subplot(2, 1, 1)
        pylab.plot(reconstructed_output)
        pylab.xlabel('Samples')
        pylab.ylabel('Amplitude')
        pylab.title('Reconstructed sentence')
        
        pylab.subplot(2, 1, 2)
        pylab.plot(original_output)
        pylab.xlabel('Samples')
        pylab.ylabel('Amplitude')
        pylab.title('Original sentence')
        
    #    pylab.subplot(3, 1, 3)
    #    pylab.plot(reconstructed_output-original_output)
    #    pylab.xlabel('Samples')
    #    pylab.ylabel('Amplitude')
    #    pylab.title('Difference')
        
        pylab.savefig(savePath+'reconstructed_data.png', format='png')
        log_file.write('\n')
        log_file.write('Reconstruction saved in '+savePath+'predicted_data.png\n')
    
        # Generate the sentence
        print '\n... ... Generating'
        for ind,k in enumerate(std_factor):
            y_gen = numpy.zeros(n_sentence_samples)
            presamples = numpy.zeros(240) #sentence_x_NSNN.get_value()[2500]
            for i in xrange(n_sentence_samples): #xrange(1000): #
                y_gen[i] = numpy.random.normal(ygen_model_NN(i,presamples.reshape((1, 240))),
                                               k*numpy.sqrt(min(train_err)/(560**2)))
                presamples = numpy.roll(presamples, -1)
                presamples[-1] = y_gen[i]
                
            output = numpy.int16(y_gen*560)
            wv.write(savePath+'Best_generated_data_'+str(ind)+'.wav', 16000, output)
        
            pylab.figure()
            pylab.plot(y_gen)
            pylab.xlabel('Samples')
            pylab.ylabel('Amplitude')
            pylab.savefig(savePath+'generated_data_'+str(ind)+'.png', format='png')
            log_file.write('Generation saved in '+savePath+'generated_data'+str(ind)+'.png \n')
        
    except:
        print('No parameters were saved.')
        log_file.write('\nNo parameters were saved. \n')

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    log_file.write('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    log_file.close()
    

if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    #theano.config.compute_test_value = 'warn'
    #theano.config.mode= 'DebugMode'
    
    path = '/home/hubert/Documents/IFT6266/TIMIT/TRAIN/DR1/FCJF0/Extracted_Features/'
#    fileNameWNN = 'win240_ARCH2_WNN_MCPM0.npz'
#    fileNameNSNN = 'win240_ARCH2_NSNN_MCPM0.npz'
    fileNameData = 'win240_ARCH2_FCJF0.npz'
    
    # Folder to save the results of the training
    savePath = '/home/hubert/Dropbox/SpeechSynthesis_Project/Results/'
    
    n_hidden_NSNN =[200,150] # for hid_NSNN in n_hidden_NSNN:
    n_hidden_WNN = [250]
    learning_rates = [0.0025]
    L2_reg = [0.0]
    
    counter = 1
    for L2 in L2_reg:
        for hid_WNN in n_hidden_WNN:
            for lamb in learning_rates:       
                
                print('Configuration '+str(counter))
                print('*****************')
                print('')
                counter += 1

                test_mlp(learning_rate=[lamb,lamb], 
                         L2_reg=[L2,L2],#[0.0001,0.0001], 
                         batch_size=1, # SHOULD STAY AT 1 ! 
                         n_epochs=50, 
                         n_hidden=[hid_WNN,n_hidden_NSNN], 
                         dataPath=path,
                         savePath=savePath,
                         fileNameData = fileNameData,
                         n_out=[2421,1])
                
                pylab.close('all')

    
    
    
    
    
    