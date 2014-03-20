# -*- coding: utf-8 -*-

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
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(layerInput, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class OutputLinear(object):
    def __init__(self, layerInput, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                               dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                               dtype=theano.config.floatX),
                               name='b', borrow=True)

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

    def __init__(self, rng, layerInput, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type layerInput: theano.tensor.TensorType
        :param layerInput: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of layerInput units, the dimension of the space in
        which the datapoints lie
        
        :type n_hidden: int
        :param n_hidden: number of hidden units
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, layerInput=layerInput,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer = OutputLinear(
            layerInput=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.outputLayer.W ** 2).sum()

        # computing the mean square errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params
        

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
        # self.showWeights()
        
    def loadParams(self, dataPath, MLPtype):
        """ Load the W1, b1, W2 and b2 parameters of the trained network
        input
            datapath: string (path to the folder with data)
            MLPtype: string (WNN or NSNN)
        """
        
        filename = 'ARCH2_'+ MLPtype +'_params.pkl' # ARCH2_WNN_params.pkl"
        f = file(dataPath+filename, 'rb')
            
        self.params = cPickle.load(f)
        f.close()
        #self.showWeights()

    def getParams(self):
        """ Get the W1, b1, W2 and b2 parameters of the MLP
        output:
            Tuple: ((W1,b1),(W2,b2))
        """
        return ((self.hiddenLayer.W.get_value(), self.hiddenLayer.b.get_value()), 
                (self.outputLayer.W.get_value(), self.outputLayer.b.get_value()))
            
            
    def setParamsFromOutput(self,y_pred):
        """ Set the W1, b1, W2 and b2 parameters of NSNN using the prediction made by WNN
        input
            y_pred: 2421 x 1 array of real numbers outputted by the WNN
        """    
		
        WNN_output = y_pred
        W1 = numpy.reshape(WNN_output[0,0:2400],(240,10))
        b1 = numpy.reshape(WNN_output[0,2400:2410],(10,))
        W2 = numpy.reshape(WNN_output[0,2410:2420],(10,1))
        b2 = numpy.reshape(WNN_output[0,2420:2421],(1,))
        
        self.hiddenLayer.W.set_value(W1, borrow=True)
        self.hiddenLayer.b.set_value(b1, borrow=True)
        self.outputLayer.W.set_value(W2, borrow=True)
        self.outputLayer.b.set_value(b2, borrow=True)
        
        
    def getTargetsFromWeights(self,weights):
        """ Set the targets of WNN using the updated weigths of NSNN
        input
            weigths: tuple of ((W1,b1),(W2,b2)), each in array form (not T.matrix!)
        """    
        
        W1 = numpy.reshape(weights[0][0],(1,-1))
        b1 = numpy.reshape(weights[0][1],(1,-1))
        W2 = numpy.reshape(weights[1][0],(1,-1))
        b2 = numpy.reshape(weights[1][1],(1,-1))
        
        return numpy.hstack((W1,b1,W2,b2))
        
    def showWeights(self):
        """This function should give a sense of the current weights of the MLP, either by printing or plotting"""
        print 'W1: ' + str(self.params[0].get_value().shape)
        print self.params[0].get_value()
        print 'b1: ' + str(self.params[1].get_value().shape)
        print self.params[1].get_value()
        print 'W2: ' + str(self.params[2].get_value().shape)
        print self.params[2].get_value()
        print 'b2: ' + str(self.params[3].get_value().shape)
        print self.params[3].get_value()        
        

def test_mlp(learning_rate=[0.15,0.15], L1_reg=[0.0,0.0], L2_reg=[0.000001,0.000001], n_epochs=1000,
             dataPath='', fileNameData='', batch_size=20, n_hidden=[5000,10], n_out=[2441,1]):
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
        print('...data was already loaded.')
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
    link_weights = 0
    link_biases = 1
    nb_in_NSNN = train_set_x_NSNN.get_value().shape[1]
    print "NSNN..."
    print "     W1: "+str(nb_hidden_units_NSNN)+" x "+str(nb_in_NSNN)    
    print "     b1: "+str(nb_hidden_units_NSNN)+" x 1"
    print "     W2: "+str(nb_out_NSNN)+" x "+str(nb_hidden_units_NSNN)    
    print "     b2: "+str(nb_out_NSNN)+" x 1"
    
    nb_in_WNN = train_set_x_WNN.get_value().shape[1]
    nb_out_WNN = 0
    if link_weights:
        nb_out_WNN = nb_hidden_units_NSNN*nb_in_NSNN + nb_out_NSNN*nb_hidden_units_NSNN
    if link_biases:
        nb_out_WNN += nb_hidden_units_NSNN + nb_out_NSNN
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
    
    # construct the WNN
    WNN = MLP(rng=rng, layerInput=x_WNN, n_in=nb_in_WNN,
                     n_hidden=nb_hidden_units_WNN, n_out=nb_out_WNN)
                     
    # the cost we minimize during training
    cost_WNN = WNN.errors(y_WNN) \
         + L1_reg[0] * WNN.L1 \
         + L2_reg[0] * WNN.L2_sqr
         
    # allocate symbolic variables for the NSNN data
    x_NSNN = T.matrix('x_NSNN')
    y_NSNN = T.dmatrix('y_NSNN')
    previous_samples = T.matrix('previous_samples_NSNN')
    index2 = T.lscalar() # index to a minibatch

    # construct the NSNN
    NSNN = MLP(rng=rng, layerInput=x_NSNN, n_in= nb_in_NSNN,
                     n_hidden=nb_hidden_units_NSNN, n_out=nb_out_NSNN)
                     
    # the cost we minimize during training
    cost_NSNN = NSNN.errors(y_NSNN) \
         + L1_reg[1] * NSNN.L1 \
         + L2_reg[1] * NSNN.L2_sqr
    
    
    # WEIGHTS NEURAL NETWORK #################################################

    # Theano expression for reshaping NSNN's weights into targets for WNN
    #WNN_targets = T.vector('WNN_targets')
    if link_weights and link_biases:
        WNN_targets = T.concatenate([T.reshape(NSNN.hiddenLayer.W,(nb_hidden_units_NSNN*nb_in_NSNN,)), 
                              T.reshape(NSNN.hiddenLayer.b,(nb_hidden_units_NSNN,)), 
                                T.reshape(NSNN.outputLayer.W,(nb_out_NSNN*nb_hidden_units_NSNN,)), 
                                T.reshape(NSNN.outputLayer.b,(nb_out_NSNN,))],
                                axis = 0)
    elif link_biases:
        WNN_targets = T.concatenate([T.reshape(NSNN.hiddenLayer.b,(nb_hidden_units_NSNN,)), 
                                T.reshape(NSNN.outputLayer.b,(nb_out_NSNN,))],
                                axis = 0)                        

    # compute the gradient of cost with respect to theta (stored in params)
    gparams_WNN = []
    for param in WNN.params:
        gparam = T.grad(cost_WNN, param)
        gparams_WNN.append(gparam)
        
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates_WNN = []
    for param, gparam in zip(WNN.params, gparams_WNN):
        updates_WNN.append((param, param - learning_rate[0] * gparam))
                
    train_model_WNN = theano.function(inputs=[index], outputs=WNN.outputLayer.y_pred,
            updates=updates_WNN,
            givens={
                x_WNN: train_set_x_WNN[index * batch_size:(index + 1) * batch_size],
                y_WNN: WNN_targets})
                
    ##########################################################################
                
                
                
    # NEXT SAMPLE NEURAL NETWORK #############################################

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model_NSNN = theano.function(inputs=[index2],
            outputs=NSNN.errors(y_NSNN),
            givens={
                x_NSNN: test_set_x_NSNN[index2 * batch_size:(index2 + 1) * batch_size],
                y_NSNN: test_set_y_NSNN[index2 * batch_size:(index2 + 1) * batch_size]})

    validate_model_NSNN = theano.function(inputs=[index2],
            outputs=NSNN.errors(y_NSNN),
            givens={
                x_NSNN: valid_set_x_NSNN[index2 * batch_size:(index2 + 1) * batch_size],
                y_NSNN: valid_set_y_NSNN[index2 * batch_size:(index2 + 1) * batch_size]})

    # compiling a Theano function that reconstructs a sentence
    yrec_model_NSNN = theano.function(inputs=[index2],
        outputs=NSNN.outputLayer.y_pred,
        givens={x_NSNN: sentence_x_NSNN[index2:index2+1]})

    # compiling a Theano function that generates the next sampleW1
    ygen_model_NSNN = theano.function(inputs=[previous_samples],
        outputs=NSNN.outputLayer.y_pred,
        givens={x_NSNN: previous_samples})

    gparams_NSNN = []
    for param in NSNN.params:
        gparam = T.grad(cost_NSNN, param)
        gparams_NSNN.append(gparam)

    updates_NSNN = []
    for param, gparam in zip(NSNN.params, gparams_NSNN):
        updates_NSNN.append((param, param - learning_rate[1] * gparam))        
    
    if link_weights and link_biases:
        W1_start = 0
        W1_end = nb_hidden_units_NSNN*nb_in_NSNN
        b1_start = W1_end
        b1_end = b1_start + nb_hidden_units_NSNN
        W2_start = b1_end
        W2_end = W2_start + nb_out_NSNN*nb_hidden_units_NSNN
        b2_start = W2_end
        b2_end = b2_start + nb_out_NSNN
        
        updates_params = range(4)    
        updates_params[0] = (NSNN.params[0], T.reshape(WNN.outputLayer.y_pred[0,W1_start:W1_end],NSNN.params[0].get_value().shape))
        updates_params[1] = (NSNN.params[1], T.reshape(WNN.outputLayer.y_pred[0,b1_start:b1_end],NSNN.params[1].get_value().shape)) 
        updates_params[2] = (NSNN.params[2], T.unbroadcast(T.reshape(WNN.outputLayer.y_pred[0,W2_start:W2_end],(nb_hidden_units_NSNN,nb_out_NSNN)),1)) 
        updates_params[3] = (NSNN.params[3], T.unbroadcast(T.reshape(WNN.outputLayer.y_pred[0,b2_start:b2_end],(nb_out_NSNN,)),0))
        
    elif link_biases:
        b1_start = 0
        b1_end = b1_start + nb_hidden_units_NSNN
        b2_start = b1_end
        b2_end = b2_start + nb_out_NSNN
    
        updates_params = range(2)    
        updates_params[0] = (NSNN.params[1], T.reshape(WNN.outputLayer.y_pred[0,b1_start:b1_end],NSNN.params[1].get_value().shape)) 
        updates_params[1] = (NSNN.params[3], T.unbroadcast(T.reshape(WNN.outputLayer.y_pred[0,b2_start:b2_end],(nb_out_NSNN,)),0))

    update_params_train_NSNN = theano.function(inputs=[index],
                        outputs=[],
                        updates = updates_params,
                        givens={x_WNN: train_set_x_WNN[index * batch_size:(index + 1) * batch_size]})
                        
    update_params_valid_NSNN = theano.function(inputs=[index],
                        outputs=[],
                        updates = updates_params,
                        givens={x_WNN: valid_set_x_WNN[index * batch_size:(index + 1) * batch_size]})
                        
    update_params_test_NSNN = theano.function(inputs=[index],
                        outputs=[],
                        updates = updates_params,
                        givens={x_WNN: test_set_x_WNN[index * batch_size:(index + 1) * batch_size]})
                
    update_params_sentence_NSNN = theano.function(inputs=[index],
                        outputs=[],
                        updates = updates_params,
                        givens={x_WNN: sentence_x_WNN[index * batch_size:(index + 1) * batch_size]})
                        
    train_model_NSNN = theano.function(inputs=[index2], outputs=cost_NSNN,
            updates=updates_NSNN,
            givens={
                x_NSNN: train_set_x_NSNN[index2 * batch_size:(index2 + 1) * batch_size],
                y_NSNN: train_set_y_NSNN[index2 * batch_size:(index2 + 1) * batch_size]})
                
    ##########################################################################

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    # Create a log to record what's happening
    date_format = '%Y%m%d%H%M%S'
    log_name = 'arch2_log_file_'+datetime.datetime.now().strftime(date_format)+'.txt'
    log_file = open(dataPath+log_name, 'w')
    log_file.write(str(datetime.datetime.now())+'\n')
    
    # Write the hyperparameters of the model
    log_file.write('Second architecture - NSNN & WNN\n')
    log_file.write('--------------------------------\n')
    log_file.write('WNN:\n')
    log_file.write('    Nb of input units: '+str(nb_in_WNN)+'\n')
    log_file.write('    Nb of hidden units: '+str(nb_hidden_units_WNN)+'\n')
    log_file.write('    Nb of output units: '+str(nb_out_WNN)+'\n')
    log_file.write('NSNN:\n')
    log_file.write('    Nb of input units: '+str(nb_in_NSNN)+'\n')
    log_file.write('    Nb of hidden units: '+str(nb_hidden_units_NSNN)+'\n')
    log_file.write('    Nb of output units: '+str(nb_out_NSNN)+'\n')
    
    # Hyperparameter values
    log_file.write('Hyperparameters (WNN, NSNN): \n')
    log_file.write('    Learning rate: '+str(learning_rate)+'\n')
    log_file.write('    L1 weight decay: '+str(L1_reg)+'\n')
    log_file.write('    L2 weight decay: '+str(L2_reg)+'\n')
    log_file.write('    Batch size: '+str(batch_size)+'\n')
    log_file.write('    Number of epochs: '+str(n_epochs)+'\n')
    
    # Data
    log_file.write('\n')
    log_file.write('Data: \n')
    log_file.write('    File: '+dataPath+fileNameData+'\n')
    log_file.write('    Number of training examples: '+str(train_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of validation examples: '+str(valid_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of test examples: '+str(test_set_y_NSNN.get_value().shape)+'\n')
    log_file.write('    Number of reconstruction/generation examples: '+str(sentence_y_NSNN.get_value().shape)+'\n')

    best_params = None
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

    epoch = 0
    done_looping = False

    log_file.write('\nTraining\n')
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        print('Epoch '+str(epoch)+':')
        
        # Training set
        for i in xrange(n_train_batches): # xrange(10000): # 
            if i%10000 == 0:
                print('    Training iteration '+str(i)+'/'+str(n_train_batches))
            train_losses[i] = (560**2)*train_model_NSNN(i)
            #NSNN.showWeights()
            #WNN.showWeights()
            train_model_WNN(i)
            #WNN.showWeights()
            update_params_train_NSNN(i)
            #NSNN.showWeights()
            #print fprop_train_WNN(i)
            #raw_input("PRESS ENTER TO CONTINUE.")
        this_train_loss = numpy.mean(train_losses)
    
        # Validation set
        for i in xrange(n_valid_batches): #xrange(100): # 
            if i%10000 == 0:
                print('    Validation iteration '+str(i)+'/'+str(n_valid_batches))
            update_params_valid_NSNN(i)
            validation_losses[i] = (560**2)*validate_model_NSNN(i)
        this_validation_loss = numpy.mean(validation_losses)
            
        # save both errors
        train_err.append(this_train_loss)
        valid_err.append(this_validation_loss)
            
        print('epoch %i, train error %f, validation error %f' %
             (epoch, this_train_loss, this_validation_loss))
        log_file.write('Epoch %i, train error %f, validation error %f' %
             (epoch, this_train_loss, this_validation_loss))
        log_file.write('\n')
        
        if math.isnan(this_train_loss) or math.isnan(this_validation_loss):
            print('Training diverged at epoch '+str(epoch))
            log_file.write('\n\nTraining diverged at epoch '+str(epoch)+'. Aborting training.')
            log_file.close()
            raise Exception("Training diverged")

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_epoch = epoch
        
            # Save the parameters of the model
            WNN.saveParams(dataPath, 'WNN')
            NSNN.saveParams(dataPath, 'NSNN')
            
#        raw_input("PRESS ENTER TO CONTINUE.")

    # Load the best model
    WNN.loadParams(dataPath, 'WNN')
    NSNN.loadParams(dataPath, 'NSNN')
    for i in xrange(n_test_batches):
        if i%10000 == 0:
            print('    Testing iteration '+str(i)+'/'+str(n_test_batches))
        update_params_test_NSNN(i)
        test_losses[i] = (560**2)*test_model_NSNN(i)
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
    pylab.savefig(dataPath+'error.png', format='png')          

    # Reconstruct the sentence
    print '... ... reconstructing'
    for i in xrange(n_sentence_samples): #xrange(1000): #
        if i%10000 == 0:
            print('    Reconstruction iteration '+str(i)+'/'+str(n_sentence_samples))
        update_params_sentence_NSNN(i)                      
        y_pred[i] = yrec_model_NSNN(i)
    
    # Save in wav format and save a figure
    reconstructed_output = numpy.int16(y_pred*560)
    wv.write(dataPath+'predicted_data.wav', 16000, reconstructed_output)
    
    original_output = numpy.int16(sentence_y_NSNN.get_value()*560)
    wv.write(dataPath+'original_data.wav', 16000, original_output)
    
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
    
    pylab.savefig(dataPath+'reconstructed_data.png', format='png')
    log_file.write('\n')
    log_file.write('Reconstruction saved in '+dataPath+'predicted_data.png\n')

    # Generate the sentence
    print '... ... Generating'
    y_gen = numpy.zeros(n_sentence_samples)
    presamples = numpy.zeros(240) #sentence_x_NSNN.get_value()[2500]
    for i in xrange(n_sentence_samples): #xrange(1000): #
        update_params_sentence_NSNN(i)
        # y_gen[i] = ygen_model_NSNN(presamples.reshape((1, 240)))
        y_gen[i] = numpy.random.normal(ygen_model_NSNN(presamples.reshape((1, 240))),
                                       numpy.sqrt(min(train_err)))
        presamples = numpy.roll(presamples, -1)
        presamples[-1] = y_gen[i]
        
    output = numpy.int16(y_gen*560)
    wv.write(dataPath+'generated_data.wav', 16000, output)
    
    pylab.figure()
    pylab.plot(y_gen)
    pylab.xlabel('Samples')
    pylab.ylabel('Amplitude')
    pylab.savefig(dataPath+'generated_data.png', format='png')
    log_file.write('Generation saved in '+dataPath+'generated_data.png \n')

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
    
    path = '/home/hubert/Documents/IFT6266/TIMIT/TRAIN/DR1/MCPM0/Extracted_Features/'
    fileNameData = 'win240_ARCH2_MCPM0.npz'
        
    test_mlp(learning_rate=[0.01,0.01], 
             L2_reg=[0.0001,0.0001], 
             batch_size=1, # SHOULD STAY AT 1 ! 
             n_epochs=30, 
             n_hidden=[250,150], 
             dataPath=path, 
             fileNameData = fileNameData,
             n_out=[2421,1])