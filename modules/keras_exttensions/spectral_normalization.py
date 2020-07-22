from keras import backend as K
from keras.engine import *
from keras import initializers
from keras.utils import conv_utils
from keras.layers import Conv2D, Conv2DTranspose
import tensorflow as tf

"""
Implementation of spectral normalization for Keras
Source: https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py
Date: 20.07.2020
"""

class ConvSN2D(Conv2D):
  def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    super(ConvSN2D, self).__init__(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)

  def build(self, input_shape):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None

    self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                             initializer=initializers.RandomNormal(0, 1),
                             name='sn',
                             trainable=False)

    # Set input spec.
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, training=None):
    def _l2normalize(v, eps=1e-12):
      return v / (K.sum(v ** 2) ** 0.5 + eps)

    def power_iteration(W, u):
      # Accroding the paper, we only need to do power iteration one time.
      _u = u
      _v = _l2normalize(K.dot(_u, K.transpose(W)))
      _u = _l2normalize(K.dot(_v, W))
      return _u, _v

    # Spectral Normalization
    W_shape = self.kernel.shape.as_list()
    # Flatten the Tensor
    W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, self.u)
    # Calculate Sigma
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    # normalize it
    W_bar = W_reshaped / sigma
    # reshape weight tensor
    if training in {0, False}:
      W_bar = K.reshape(W_bar, W_shape)
    else:
      with tf.control_dependencies([self.u.assign(_u)]):
        W_bar = K.reshape(W_bar, W_shape)

    outputs = K.conv2d(
      inputs,
      W_bar,
      strides=self.strides,
      padding=self.padding,
      data_format=self.data_format,
      dilation_rate=self.dilation_rate)
    if self.use_bias:
      outputs = K.bias_add(
        outputs,
        self.bias,
        data_format=self.data_format)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

class ConvSN2DTranspose(Conv2DTranspose):
  def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    super(ConvSN2DTranspose, self).__init__(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank ' + str(4) +
                       '; Received input shape:', str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None

    self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                             initializer=initializers.RandomNormal(0, 1),
                             name='sn',
                             trainable=False)

    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    input_shape = K.shape(inputs)
    batch_size = input_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides
    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_length(height,
                                          stride_h, kernel_h,
                                          self.padding,
                                          out_pad_h)
    out_width = conv_utils.deconv_length(width,
                                         stride_w, kernel_w,
                                         self.padding,
                                         out_pad_w)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    # Spectral Normalization
    def _l2normalize(v, eps=1e-12):
      return v / (K.sum(v ** 2) ** 0.5 + eps)

    def power_iteration(W, u):
      # Accroding the paper, we only need to do power iteration one time.
      _u = u
      _v = _l2normalize(K.dot(_u, K.transpose(W)))
      _u = _l2normalize(K.dot(_v, W))
      return _u, _v

    W_shape = self.kernel.shape.as_list()
    # Flatten the Tensor
    W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, self.u)
    # Calculate Sigma
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    # normalize it
    W_bar = W_reshaped / sigma
    # reshape weight tensor
    if training in {0, False}:
      W_bar = K.reshape(W_bar, W_shape)
    else:
      with tf.control_dependencies([self.u.assign(_u)]):
        W_bar = K.reshape(W_bar, W_shape)
    self.kernel = W_bar

    outputs = K.conv2d_transpose(
      inputs,
      self.kernel,
      output_shape,
      self.strides,
      padding=self.padding,
      data_format=self.data_format)

    if self.use_bias:
      outputs = K.bias_add(
        outputs,
        self.bias,
        data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs