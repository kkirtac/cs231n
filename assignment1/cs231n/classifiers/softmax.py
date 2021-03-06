import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_dim = W.shape[0]
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  p = np.zeros((num_train, num_classes))  #class probabilities
  yy = np.zeros((num_train, num_classes)) #labels


  for i in xrange(num_train):

    f = X[i,:].dot(W)

    f -= np.max(f)   #regularization to avoid numeric instability

    normalization_term = np.sum(np.exp(f))

    p[i,:] = np.exp(f) / normalization_term

    yy[i,y[i]] = 1

    loss += - np.sum( yy[i,:] * np.log( p[i,:] ) )
    
    grad = X[i,:].reshape(num_dim,1).dot( (p[i,:] - yy[i,:]).reshape(1,num_classes) )
    
    dW = dW + grad

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  dW += reg*W


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_dim, num_classes = W.shape
  num_train = X.shape[0]
    
  p = np.zeros((num_train, num_classes))  #class probabilities
  yy = np.zeros((num_train, num_classes)) #labels

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  f = X.dot(W)
  f -= np.amax(f,axis=1).reshape(num_train,1)   #regularization to avoid numeric instability
  normalization_term = np.sum(np.exp(f), axis=1).reshape(num_train,1)
  p = np.exp(f) / normalization_term

  yy[np.arange(num_train),y] = 1
  
  loss += - np.sum( yy * np.log( p ) )   
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) 

  dW = dW + X.T.dot(p-yy)
    
  dW /= num_train

  dW += reg*W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

