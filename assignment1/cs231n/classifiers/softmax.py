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
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[1]  
  num_train = X.shape[0]
  d_score = np.zeros((num_train, num_class))
  for i in range(num_train):
    score = X[i].dot(W)
    score -= np.max(score) # for numeric stability
    exponential = np.exp(score)
    probability_norm = exponential / np.sum(exponential)
    loss -= np.log(probability_norm[y[i]])
    probability_norm[y[i]] -= 1 # d_score : 1 x C
    d_score[i] = probability_norm

  loss /= num_train
  loss += reg * np.sum(W ** 2)
  
  dW = X.T.dot(d_score)
  dW /= num_train
  dW += 2 * reg * W
  
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  score = X.dot(W)
  score -= np.max(score, axis = 1).reshape(-1, 1) # for numerical stability
  exp_score = np.exp(score)
  probability_norm = exp_score / np.sum(exp_score, axis=1).reshape(-1, 1)

  loss = -1 * np.sum(np.log(probability_norm[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W ** 2)

  probability_norm[np.arange(num_train), y] -= 1 # d_score
  dW = X.T.dot(probability_norm)
  dW /= num_train
  dW += 2 * reg * W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

