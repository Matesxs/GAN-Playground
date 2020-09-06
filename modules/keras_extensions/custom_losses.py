import keras.backend as K
import numpy as np

def wasserstein_loss(y_true, y_pred):
  return K.mean(y_true * y_pred)

def gradient_penalty_loss(_, y_pred, averaged_samples):
  gradients = K.gradients(y_pred, averaged_samples)[0]
  gradients_sqr = K.square(gradients)
  gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
  gradient_l2_norm = K.sqrt(gradients_sqr_sum)
  gradient_penalty = K.square(1 - gradient_l2_norm)
  return K.mean(gradient_penalty)