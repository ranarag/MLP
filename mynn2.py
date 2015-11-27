# -*- coding: utf-8 -*-
###########################################
#       IMPORT MODULES
###########################################
import random
import numpy as np


###########################################
#       TRANSFER FUNCTIONS
###########################################
def sigmoid(z, Derivative = False):
    """The sigmoid function."""
    if not Derivative:
        return 1.0/(1.0+np.exp(-z))
    else:
        out = sigmoid(z)
        return out*(1-out)
    #return np.tanh(z)

def linear( x, Derivative = False):
        if not Derivative:
           return x
            #return np.tanh(x)
        else:
            return 1.0
def gaussian( x, Derivative = False):
        if not Derivative:
            return np.exp(-x**2)
            #return np.tanh(x)
        else:
            
            return -2*x*np.exp(-x**2)
def tanh(x, Derivative = False):
        if not Derivative:
             return np.tanh(x)
            
        else:
            return 1.0 - np.tanh(x)**2


###########################################
#         NN CLASS
###########################################
class Network(object):
    '''init function'''
    def __init__(self, sizes,tf):
        
        self.tf = tf
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    ''' for feedforward'''    
    def feedforward(self, a):
        i = 0
        for b, w in zip(self.biases, self.weights):
            #print str(w)
            i=i+1
            a = self.tf[i](w.dot(a) + b)
            
            #print "ok"+str(a)
        return a
            
    ''' stochastic gradient descent'''
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        
        
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
    def update_mini_batch(self, mini_batch, eta):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    ''' backpropagation function'''
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] 
        zs = [] 
        index = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            index += 1
            activation = self.tf[index](z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * self.tf[-1](zs[-1],True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        k = self.cost_derivative(activations[-1], y)
        print "error = " +str(np.sum(k**2))
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.tf[-l](z,True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    ''' evaluate function'''
    def evaluate(self, test_data):
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    ''' for running test cases'''
    def test_run(self, x,predict=True):
        if predict:
            return np.sum(self.feedforward(x))
        else:
            return np.argmax(self.feedforward(x))
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    
if __name__=="__main__":
    lFuncs = [sigmoid,gaussian,linear]
    net = Network([2,2,1],lFuncs)
    a = np.array([[0,0],[0,1],[1,0],[1,1]])
    k = [np.reshape(x ,(2,1) ) for x in a]
    y = np.array([[0],[1],[1],[0]])
    v = [x for x in y]
    train = zip(k ,y)
    net.SGD(train,1000,1,0.2)
    m =net.test_run(k[0])
    print str(m)
    m =net.test_run(k[1])
    print str(m)
    m =net.test_run(k[2])
    print str(m)
    m =net.test_run(k[3])
    print str(m)    
    
                
    
    
         