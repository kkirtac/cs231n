import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
    
  from pprint import pprint

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):  
    count = 0
    scores = X[i,:].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j] = dW[:,j] + X[i,:]
        loss += margin
        count += 1
    dW[:,y[i]] = dW[:,y[i]] - count * X[i,:]

  dW /= num_train

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):

  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train, num_dim = X.shape
  num_classes = W.shape[1]
    
  from pprint import pprint

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)                                                          # NxC
  correct_class_score = scores[np.arange(num_train),y]                       # Nx1
    
  margin = scores - correct_class_score.reshape((num_train,1)) + 1.0         # NxC

  margin[np.arange(num_train),y] = 0
    
  count = np.sum(margin > 0, axis=1).reshape((num_train,1))                  # Nx1 
  indices = margin > 0                                                       # NxC

  loss = np.sum(margin) / num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
   
 
  aa = count * X
  for j in xrange(num_classes):
    idx = y == j
    k = np.sum(aa[idx], axis=0)
    dW[:,j] = dW[:,j] - k.T;
    
    idx = indices[:,j] & (~(y == j))
    k = np.sum(X[idx], axis=0)
    dW[:,j] = dW[:,j] + k.T;
    
  dW /= num_train

  dW += reg*W
    
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
